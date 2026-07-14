use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Duration;

use rustc_target::spec::Os;

use crate::shims::files::FdNum;
use crate::*;

/// An interest into a file descriptor together with its
/// relevant readiness events.
#[derive(Debug)]
struct PollInterest<'tcx> {
    /// Place where the ready events of the interests should be written to.
    revents_place: MPlaceTy<'tcx>,
}

impl VisitProvenance for PollInterest<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        self.revents_place.visit_provenance(visit);
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn poll(
        &mut self,
        fds: &OpTy<'tcx>,
        nfds: &OpTy<'tcx>,
        timeout: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let nfds_layout = this.libc_ty_layout("nfds_t");
        let nfds: u64 = this.read_scalar(nfds)?.to_int(nfds_layout.size)?.try_into().unwrap();
        let timeout = this.read_scalar(timeout)?.to_i32()?;
        let fds_arr_layout = this.libc_array_ty_layout("pollfd", nfds);
        let fds_arr_mplace = this.deref_pointer_as(fds, fds_arr_layout)?;
        let mut fds_arr_iter = this.project_array_fields(&fds_arr_mplace)?;

        // The provided interests indexed by the file descriptor they're for.
        let mut interests = BTreeMap::<FdNum, PollInterest<'tcx>>::new();
        // Counts the number of poll interests that are invalid because they're for
        // a positive file descriptor that doesn't exist.
        let mut invalid_interests = 0u32;

        let watcher = Rc::new(this.machine.readiness_interests.new_watcher());

        // We iterate over the fds array of the `poll` syscall. For each fd, we check its
        // output field, the relevant events, and whether they are currently fulfilled.
        while let Some((_idx, pollfd)) = fds_arr_iter.next(this)? {
            let fd_field = this.project_field_named(&pollfd, "fd")?;
            let fd_num = this.read_scalar(&fd_field)?.to_i32()?;
            let events_field = this.project_field_named(&pollfd, "events")?;
            let events = this.read_scalar(&events_field)?.to_u16()?;
            let revents_field = this.project_field_named(&pollfd, "revents")?;

            let relevant_events = this.poll_bitflag_to_readiness(events)?;

            let revents = if this.machine.fds.get(fd_num).is_some() {
                // A file description for this file descriptor exists; the interest is thus not ignored.
                let interest = PollInterest { revents_place: revents_field.clone() };
                if interests.try_insert(fd_num, interest).is_err() {
                    throw_unsup_format!(
                        "poll: providing multiple interests for the same file descriptor is unsupported"
                    )
                }
                watcher
                    .add_interest(
                        fd_num,
                        relevant_events,
                        /* is_edge_triggered */ false,
                        u64::try_from(fd_num).unwrap(),
                        this,
                    )?
                    // We just ensured that no file descriptor is registered twice.
                    .unwrap();

                // Since we later only update the `revents` field for FDs which receive
                // an event, we initially zero this field.
                0
            } else if fd_num.is_negative() {
                // Interests for negative file descriptors should be ignored and
                // their `revents` field should be zeroed.
                0
            } else {
                // Interests for positive, invalid file descriptors should be ignored
                // and their `revents` field should be set to POLLNVAL.

                // The Linux implementation still counts such interests as "fulfilled"
                // and thus returns from the `poll` invocation.
                invalid_interests = invalid_interests.strict_add(1);

                this.eval_libc_u16("POLLNVAL")
            };

            this.write_scalar(Scalar::from_u16(revents), &revents_field)?;
        }

        if timeout == 0 || invalid_interests > 0 || watcher.ready_count() > 0 {
            // Some interests are already fulfilled or a zero timeout was provided.
            // We thus don't need to block the thread and can just return here.

            let count = this.write_ready_events(watcher, interests)?;
            // The Linux implementation also counts invalid interests as fulfilled.
            let total = count.strict_add(invalid_interests);

            return this.write_scalar(Scalar::from_u32(total), dest);
        }

        // None of the interests are currently fulfilled; we thus need to
        // block the thread until any interest gets fulfilled.
        watcher.add_blocked_thread(this.machine.threads.active_thread());

        let deadline = if timeout.is_positive() {
            let timeout_duration = Duration::from_millis(u64::try_from(timeout).unwrap());
            Some(this.machine.monotonic_clock.now().add_lossy(timeout_duration).into())
        } else {
            // Negative timeout means block indefinitely.
            None
        };

        let dest = dest.clone();
        this.block_thread(
            BlockReason::Readiness,
            deadline,
            callback!(
                @capture<'tcx> {
                    watcher: Rc<ReadinessWatcher>,
                    interests: BTreeMap<FdNum, PollInterest<'tcx>>,
                    dest: MPlaceTy<'tcx>,
                } |this, reason: UnblockKind| {
                    if let UnblockKind::TimedOut = reason {
                        return this.write_scalar(Scalar::from_u32(0), &dest);
                    }

                    let count = this.write_ready_events(watcher, interests)?;
                    this.write_scalar(Scalar::from_u32(count), &dest)
                }
            ),
        );

        interp_ok(())
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// For all ready interests on the watcher, write the appropriate
    /// readiness into the `revents` field of the associated poll interest.
    fn write_ready_events(
        &mut self,
        watcher: Rc<ReadinessWatcher>,
        interests: BTreeMap<FdNum, PollInterest<'tcx>>,
    ) -> InterpResult<'tcx, u32> {
        let this = self.eval_context_mut();

        // Counts the number of poll interests that are fulfilled.
        let mut fulfilled_interests = 0u32;

        // Iterate over all ready interests of the watcher and
        // write the output readiness of all related poll interests.
        for ready in watcher.get_ready_interests(watcher.ready_count(), this)? {
            let fd_num = FdNum::try_from(ready.data).expect("Data is always a file descriptor");
            let interest = interests.get(&fd_num).expect("Interest should exist");

            fulfilled_interests = fulfilled_interests.strict_add(1);

            let poll_events = this.readiness_to_poll_bitflag(ready.active());
            this.write_scalar(Scalar::from_u16(poll_events), &interest.revents_place)?;
        }

        interp_ok(fulfilled_interests)
    }

    /// Convert a [`Readiness`] instance into the corresponding poll
    /// readiness bitflag.
    fn readiness_to_poll_bitflag(&self, readiness: &Readiness) -> u16 {
        let this = self.eval_context_ref();

        let pollin = this.eval_libc_u16("POLLIN");
        let pollout = this.eval_libc_u16("POLLOUT");
        let pollhup = this.eval_libc_u16("POLLHUP");
        let pollerr = this.eval_libc_u16("POLLERR");

        let mut bitflag = 0;
        if readiness.readable {
            bitflag |= pollin;
        }
        if readiness.writable {
            bitflag |= pollout;
        }
        if readiness.write_closed {
            bitflag |= pollhup;
        }
        if readiness.error {
            bitflag |= pollerr;
        }

        if matches!(this.tcx.sess.target.os, Os::Linux | Os::Android | Os::FreeBsd | Os::Illumos) {
            // POLLRDHUP only exists on Linux, Android, FreeBSD, and Illumos.
            let pollrdhup = this.eval_libc_u16("POLLRDHUP");
            if readiness.read_closed {
                bitflag |= pollrdhup;
            }
        }

        bitflag
    }

    /// Convert a poll readiness bitflag into the corresponding [`Readiness`] instance.
    ///
    /// This always sets the `write_closed` and `error` readiness since they are
    /// implicitly registered for any interest with `poll`.
    fn poll_bitflag_to_readiness(&self, mut bitflag: u16) -> InterpResult<'tcx, Readiness> {
        let this = self.eval_context_ref();

        let pollin = this.eval_libc_u16("POLLIN");
        let pollout = this.eval_libc_u16("POLLOUT");
        let pollhup = this.eval_libc_u16("POLLHUP");
        let pollerr = this.eval_libc_u16("POLLERR");
        let pollnval = this.eval_libc_u16("POLLNVAL");

        // The POLLHUP and POLLERR interests are always set.
        let mut readiness = Readiness { write_closed: true, error: true, ..Readiness::EMPTY };

        if bitflag & pollin == pollin {
            readiness.readable = true;
            bitflag &= !pollin;
        }
        if bitflag & pollout == pollout {
            readiness.writable = true;
            bitflag &= !pollout;
        }
        if bitflag & pollhup == pollhup {
            bitflag &= !pollhup;
        }
        if bitflag & pollerr == pollerr {
            bitflag &= !pollerr;
        }
        if bitflag & pollnval == pollnval {
            // POLLNVAL is ignored when it's provided as a relevant event.
            bitflag &= !pollnval;
        }

        if matches!(this.tcx.sess.target.os, Os::Linux | Os::Android | Os::FreeBsd | Os::Illumos) {
            // POLLRDHUP only exists on Linux, Android, FreeBSD, and Illumos.
            let pollrdhup = this.eval_libc_u16("POLLRDHUP");
            if bitflag & pollrdhup == pollrdhup {
                readiness.read_closed = true;
                bitflag &= !pollrdhup;
            }
        }

        if bitflag != 0 {
            throw_unsup_format!(
                "poll: poll event {bitflag:#x} is unsupported. Only POLLIN, \
                POLLOUT, POLLERR, POLLHUP, POLLNVAL and POLLRDHUP are supported."
            );
        }

        interp_ok(readiness)
    }
}
