/// Oneshot channels/ports
///
/// This is the initial flavor of channels/ports used for comm module. This is
/// an optimization for the one-use case of a channel. The major optimization of
/// this type is to have one and exactly one allocation when the chan/port pair
/// is created.
///
/// Another possible optimization would be to not use an Arc box because
/// in theory we know when the shared packet can be deallocated (no real need
/// for the atomic reference counting), but I was having trouble how to destroy
/// the data early in a drop of a Port.
///
/// # Implementation
///
/// Oneshots are implemented around one atomic usize variable. This variable
/// indicates both the state of the port/chan but also contains any threads
/// blocked on the port. All atomic operations happen on this one word.
///
/// In order to upgrade a oneshot channel, an upgrade is considered a disconnect
/// on behalf of the channel side of things (it can be mentally thought of as
/// consuming the port). This upgrade is then also stored in the shared packet.
/// The one caveat to consider is that when a port sees a disconnected channel
/// it must check for data because there is no "data plus upgrade" state.
pub use self::Failure::*;
use self::MyUpgrade::*;
pub use self::UpgradeResult::*;

use crate::cell::UnsafeCell;
use crate::ptr;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::mpsc::blocking::{self, SignalToken};
use crate::sync::mpsc::Receiver;
use crate::time::Instant;

// Various states you can find a port in.
const EMPTY: usize = 0; // initial state: no data, no blocked receiver
const DATA: usize = 1; // data ready for receiver to take
const DISCONNECTED: usize = 2; // channel is disconnected OR upgraded
// Any other value represents a pointer to a SignalToken value. The
// protocol ensures that when the state moves *to* a pointer,
// ownership of the token is given to the packet, and when the state
// moves *from* a pointer, ownership of the token is transferred to
// whoever changed the state.

pub struct Packet<T> {
    // Internal state of the chan/port pair (stores the blocked thread as well)
    state: AtomicUsize,
    // One-shot data slot location
    data: UnsafeCell<Option<T>>,
    // when used for the second time, a oneshot channel must be upgraded, and
    // this contains the slot for the upgrade
    upgrade: UnsafeCell<MyUpgrade<T>>,
}

pub enum Failure<T> {
    Empty,
    Disconnected,
    Upgraded(Receiver<T>),
}

pub enum UpgradeResult {
    UpSuccess,
    UpDisconnected,
    UpWoke(SignalToken),
}

enum MyUpgrade<T> {
    NothingSent,
    SendUsed,
    GoUp(Receiver<T>),
}

impl<T> Packet<T> {
    pub fn new() -> Packet<T> {
        Packet {
            data: UnsafeCell::new(None),
            upgrade: UnsafeCell::new(NothingSent),
            state: AtomicUsize::new(EMPTY),
        }
    }

    pub fn send(&self, t: T) -> Result<(), T> {
        unsafe {
            // Sanity check
            match *self.upgrade.get() {
                NothingSent => {}
                _ => panic!("sending on a oneshot that's already sent on "),
            }
            assert!((*self.data.get()).is_none());
            ptr::write(self.data.get(), Some(t));
            ptr::write(self.upgrade.get(), SendUsed);

            match self.state.swap(DATA, Ordering::SeqCst) {
                // Sent the data, no one was waiting
                EMPTY => Ok(()),

                // Couldn't send the data, the port hung up first. Return the data
                // back up the stack.
                DISCONNECTED => {
                    self.state.swap(DISCONNECTED, Ordering::SeqCst);
                    ptr::write(self.upgrade.get(), NothingSent);
                    Err((&mut *self.data.get()).take().unwrap())
                }

                // Not possible, these are one-use channels
                DATA => unreachable!(),

                // There is a thread waiting on the other end. We leave the 'DATA'
                // state inside so it'll pick it up on the other end.
                ptr => {
                    SignalToken::cast_from_usize(ptr).signal();
                    Ok(())
                }
            }
        }
    }

    // Just tests whether this channel has been sent on or not, this is only
    // safe to use from the sender.
    pub fn sent(&self) -> bool {
        unsafe { !matches!(*self.upgrade.get(), NothingSent) }
    }

    pub fn recv(&self, deadline: Option<Instant>) -> Result<T, Failure<T>> {
        // Attempt to not block the thread (it's a little expensive). If it looks
        // like we're not empty, then immediately go through to `try_recv`.
        if self.state.load(Ordering::SeqCst) == EMPTY {
            let (wait_token, signal_token) = blocking::tokens();
            let ptr = unsafe { signal_token.cast_to_usize() };

            // race with senders to enter the blocking state
            if self.state.compare_exchange(EMPTY, ptr, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                if let Some(deadline) = deadline {
                    let timed_out = !wait_token.wait_max_until(deadline);
                    // Try to reset the state
                    if timed_out {
                        self.abort_selection().map_err(Upgraded)?;
                    }
                } else {
                    wait_token.wait();
                    debug_assert!(self.state.load(Ordering::SeqCst) != EMPTY);
                }
            } else {
                // drop the signal token, since we never blocked
                drop(unsafe { SignalToken::cast_from_usize(ptr) });
            }
        }

        self.try_recv()
    }

    pub fn try_recv(&self) -> Result<T, Failure<T>> {
        unsafe {
            match self.state.load(Ordering::SeqCst) {
                EMPTY => Err(Empty),

                // We saw some data on the channel, but the channel can be used
                // again to send us an upgrade. As a result, we need to re-insert
                // into the channel that there's no data available (otherwise we'll
                // just see DATA next time). This is done as a cmpxchg because if
                // the state changes under our feet we'd rather just see that state
                // change.
                DATA => {
                    let _ = self.state.compare_exchange(
                        DATA,
                        EMPTY,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    );
                    match (&mut *self.data.get()).take() {
                        Some(data) => Ok(data),
                        None => unreachable!(),
                    }
                }

                // There's no guarantee that we receive before an upgrade happens,
                // and an upgrade flags the channel as disconnected, so when we see
                // this we first need to check if there's data available and *then*
                // we go through and process the upgrade.
                DISCONNECTED => match (&mut *self.data.get()).take() {
                    Some(data) => Ok(data),
                    None => match ptr::replace(self.upgrade.get(), SendUsed) {
                        SendUsed | NothingSent => Err(Disconnected),
                        GoUp(upgrade) => Err(Upgraded(upgrade)),
                    },
                },

                // We are the sole receiver; there cannot be a blocking
                // receiver already.
                _ => unreachable!(),
            }
        }
    }

    // Returns whether the upgrade was completed. If the upgrade wasn't
    // completed, then the port couldn't get sent to the other half (it will
    // never receive it).
    pub fn upgrade(&self, up: Receiver<T>) -> UpgradeResult {
        unsafe {
            let prev = match *self.upgrade.get() {
                NothingSent => NothingSent,
                SendUsed => SendUsed,
                _ => panic!("upgrading again"),
            };
            ptr::write(self.upgrade.get(), GoUp(up));

            match self.state.swap(DISCONNECTED, Ordering::SeqCst) {
                // If the channel is empty or has data on it, then we're good to go.
                // Senders will check the data before the upgrade (in case we
                // plastered over the DATA state).
                DATA | EMPTY => UpSuccess,

                // If the other end is already disconnected, then we failed the
                // upgrade. Be sure to trash the port we were given.
                DISCONNECTED => {
                    ptr::replace(self.upgrade.get(), prev);
                    UpDisconnected
                }

                // If someone's waiting, we gotta wake them up
                ptr => UpWoke(SignalToken::cast_from_usize(ptr)),
            }
        }
    }

    pub fn drop_chan(&self) {
        match self.state.swap(DISCONNECTED, Ordering::SeqCst) {
            DATA | DISCONNECTED | EMPTY => {}

            // If someone's waiting, we gotta wake them up
            ptr => unsafe {
                SignalToken::cast_from_usize(ptr).signal();
            },
        }
    }

    pub fn drop_port(&self) {
        match self.state.swap(DISCONNECTED, Ordering::SeqCst) {
            // An empty channel has nothing to do, and a remotely disconnected
            // channel also has nothing to do b/c we're about to run the drop
            // glue
            DISCONNECTED | EMPTY => {}

            // There's data on the channel, so make sure we destroy it promptly.
            // This is why not using an arc is a little difficult (need the box
            // to stay valid while we take the data).
            DATA => unsafe {
                (&mut *self.data.get()).take().unwrap();
            },

            // We're the only ones that can block on this port
            _ => unreachable!(),
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // select implementation
    ////////////////////////////////////////////////////////////////////////////

    // Remove a previous selecting thread from this port. This ensures that the
    // blocked thread will no longer be visible to any other threads.
    //
    // The return value indicates whether there's data on this port.
    pub fn abort_selection(&self) -> Result<bool, Receiver<T>> {
        let state = match self.state.load(Ordering::SeqCst) {
            // Each of these states means that no further activity will happen
            // with regard to abortion selection
            s @ (EMPTY | DATA | DISCONNECTED) => s,

            // If we've got a blocked thread, then use an atomic to gain ownership
            // of it (may fail)
            ptr => self
                .state
                .compare_exchange(ptr, EMPTY, Ordering::SeqCst, Ordering::SeqCst)
                .unwrap_or_else(|x| x),
        };

        // Now that we've got ownership of our state, figure out what to do
        // about it.
        match state {
            EMPTY => unreachable!(),
            // our thread used for select was stolen
            DATA => Ok(true),

            // If the other end has hung up, then we have complete ownership
            // of the port. First, check if there was data waiting for us. This
            // is possible if the other end sent something and then hung up.
            //
            // We then need to check to see if there was an upgrade requested,
            // and if so, the upgraded port needs to have its selection aborted.
            DISCONNECTED => unsafe {
                if (*self.data.get()).is_some() {
                    Ok(true)
                } else {
                    match ptr::replace(self.upgrade.get(), SendUsed) {
                        GoUp(port) => Err(port),
                        _ => Ok(true),
                    }
                }
            },

            // We woke ourselves up from select.
            ptr => unsafe {
                drop(SignalToken::cast_from_usize(ptr));
                Ok(false)
            },
        }
    }
}

impl<T> Drop for Packet<T> {
    fn drop(&mut self) {
        assert_eq!(self.state.load(Ordering::SeqCst), DISCONNECTED);
    }
}
