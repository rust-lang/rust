// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Ports and channels.

use option::*;
use cast;
use ops::Drop;
use rt::kill::BlockedTask;
use kinds::Send;
use rt::sched::Scheduler;
use rt::local::Local;
use rt::select::{Select, SelectPort};
use unstable::atomics::{AtomicUint, AtomicOption, Acquire, Release, SeqCst};
use unstable::sync::UnsafeAtomicRcBox;
use util::Void;
use comm::{GenericChan, GenericSmartChan, GenericPort, Peekable};
use cell::Cell;
use clone::Clone;

/// A combined refcount / BlockedTask-as-uint pointer.
///
/// Can be equal to the following values:
///
/// * 2 - both endpoints are alive
/// * 1 - either the sender or the receiver is dead, determined by context
/// * <ptr> - A pointer to a blocked Task (see BlockedTask::cast_{to,from}_uint)
type State = uint;

static STATE_BOTH: State = 2;
static STATE_ONE: State = 1;

/// The heap-allocated structure shared between two endpoints.
struct Packet<T> {
    state: AtomicUint,
    payload: Option<T>,
}

/// A one-shot channel.
pub struct ChanOne<T> {
    void_packet: *mut Void,
    suppress_finalize: bool
}

/// A one-shot port.
pub struct PortOne<T> {
    void_packet: *mut Void,
    suppress_finalize: bool
}

pub fn oneshot<T: Send>() -> (PortOne<T>, ChanOne<T>) {
    let packet: ~Packet<T> = ~Packet {
        state: AtomicUint::new(STATE_BOTH),
        payload: None
    };

    unsafe {
        let packet: *mut Void = cast::transmute(packet);
        let port = PortOne {
            void_packet: packet,
            suppress_finalize: false
        };
        let chan = ChanOne {
            void_packet: packet,
            suppress_finalize: false
        };
        return (port, chan);
    }
}

impl<T> ChanOne<T> {
    #[inline]
    fn packet(&self) -> *mut Packet<T> {
        unsafe {
            let p: *mut ~Packet<T> = cast::transmute(&self.void_packet);
            let p: *mut Packet<T> = &mut **p;
            return p;
        }
    }

    pub fn send(self, val: T) {
        self.try_send(val);
    }

    pub fn try_send(self, val: T) -> bool {
        let mut this = self;
        let mut recvr_active = true;
        let packet = this.packet();

        unsafe {

            // Install the payload
            assert!((*packet).payload.is_none());
            (*packet).payload = Some(val);

            // Atomically swap out the old state to figure out what
            // the port's up to, issuing a release barrier to prevent
            // reordering of the payload write. This also issues an
            // acquire barrier that keeps the subsequent access of the
            // ~Task pointer from being reordered.
            let oldstate = (*packet).state.swap(STATE_ONE, SeqCst);
            match oldstate {
                STATE_BOTH => {
                    // Port is not waiting yet. Nothing to do
                    do Local::borrow::<Scheduler, ()> |sched| {
                        rtdebug!("non-rendezvous send");
                        sched.metrics.non_rendezvous_sends += 1;
                    }
                }
                STATE_ONE => {
                    do Local::borrow::<Scheduler, ()> |sched| {
                        rtdebug!("rendezvous send");
                        sched.metrics.rendezvous_sends += 1;
                    }
                    // Port has closed. Need to clean up.
                    let _packet: ~Packet<T> = cast::transmute(this.void_packet);
                    recvr_active = false;
                }
                task_as_state => {
                    // Port is blocked. Wake it up.
                    let recvr = BlockedTask::cast_from_uint(task_as_state);
                    do recvr.wake().map_consume |woken_task| {
                        let mut sched = Local::take::<Scheduler>();
                        rtdebug!("rendezvous send");
                        sched.metrics.rendezvous_sends += 1;
                        sched.schedule_task(woken_task);
                    };
                }
            }
        }

        // Suppress the synchronizing actions in the finalizer. We're done with the packet.
        this.suppress_finalize = true;
        return recvr_active;
    }
}

impl<T> PortOne<T> {
    fn packet(&self) -> *mut Packet<T> {
        unsafe {
            let p: *mut ~Packet<T> = cast::transmute(&self.void_packet);
            let p: *mut Packet<T> = &mut **p;
            return p;
        }
    }

    pub fn recv(self) -> T {
        match self.try_recv() {
            Some(val) => val,
            None => {
                fail!("receiving on closed channel");
            }
        }
    }

    pub fn try_recv(self) -> Option<T> {
        let mut this = self;

        // Optimistic check. If data was sent already, we don't even need to block.
        // No release barrier needed here; we're not handing off our task pointer yet.
        if !this.optimistic_check() {
            // No data available yet.
            // Switch to the scheduler to put the ~Task into the Packet state.
            let sched = Local::take::<Scheduler>();
            do sched.deschedule_running_task_and_then |sched, task| {
                this.block_on(sched, task);
            }
        }

        // Task resumes.
        this.recv_ready()
    }
}

impl<T> Select for PortOne<T> {
    #[inline] #[cfg(not(test))]
    fn optimistic_check(&mut self) -> bool {
        unsafe { (*self.packet()).state.load(Acquire) == STATE_ONE }
    }

    #[inline] #[cfg(test)]
    fn optimistic_check(&mut self) -> bool {
        // The optimistic check is never necessary for correctness. For testing
        // purposes, making it randomly return false simulates a racing sender.
        use rand::{Rand, rng};
        let mut rng = rng();
        let actually_check = Rand::rand(&mut rng);
        if actually_check {
            unsafe { (*self.packet()).state.load(Acquire) == STATE_ONE }
        } else {
            false
        }
    }

    fn block_on(&mut self, sched: &mut Scheduler, task: BlockedTask) -> bool {
        unsafe {
            // Atomically swap the task pointer into the Packet state, issuing
            // an acquire barrier to prevent reordering of the subsequent read
            // of the payload. Also issues a release barrier to prevent
            // reordering of any previous writes to the task structure.
            let task_as_state = task.cast_to_uint();
            let oldstate = (*self.packet()).state.swap(task_as_state, SeqCst);
            match oldstate {
                STATE_BOTH => {
                    // Data has not been sent. Now we're blocked.
                    rtdebug!("non-rendezvous recv");
                    sched.metrics.non_rendezvous_recvs += 1;
                    false
                }
                STATE_ONE => {
                    // Re-record that we are the only owner of the packet.
                    // Release barrier needed in case the task gets reawoken
                    // on a different core (this is analogous to writing a
                    // payload; a barrier in enqueueing the task protects it).
                    // NB(#8132). This *must* occur before the enqueue below.
                    // FIXME(#6842, #8130) This is usually only needed for the
                    // assertion in recv_ready, except in the case of select().
                    // This won't actually ever have cacheline contention, but
                    // maybe should be optimized out with a cfg(test) anyway?
                    (*self.packet()).state.store(STATE_ONE, Release);

                    rtdebug!("rendezvous recv");
                    sched.metrics.rendezvous_recvs += 1;

                    // Channel is closed. Switch back and check the data.
                    // NB: We have to drop back into the scheduler event loop here
                    // instead of switching immediately back or we could end up
                    // triggering infinite recursion on the scheduler's stack.
                    let recvr = BlockedTask::cast_from_uint(task_as_state);
                    sched.enqueue_blocked_task(recvr);
                    true
                }
                _ => rtabort!("can't block_on; a task is already blocked")
            }
        }
    }

    // This is the only select trait function that's not also used in recv.
    fn unblock_from(&mut self) -> bool {
        let packet = self.packet();
        unsafe {
            // In case the data is available, the acquire barrier here matches
            // the release barrier the sender used to release the payload.
            match (*packet).state.load(Acquire) {
                // Impossible. We removed STATE_BOTH when blocking on it, and
                // no self-respecting sender would put it back.
                STATE_BOTH    => rtabort!("refcount already 2 in unblock_from"),
                // Here, a sender already tried to wake us up. Perhaps they
                // even succeeded! Data is available.
                STATE_ONE     => true,
                // Still registered as blocked. Need to "unblock" the pointer.
                task_as_state => {
                    // In the window between the load and the CAS, a sender
                    // might take the pointer and set the refcount to ONE. If
                    // that happens, we shouldn't clobber that with BOTH!
                    // Acquire barrier again for the same reason as above.
                    match (*packet).state.compare_and_swap(task_as_state, STATE_BOTH,
                                                           Acquire) {
                        STATE_BOTH => rtabort!("refcount became 2 in unblock_from"),
                        STATE_ONE  => true, // Lost the race. Data available.
                        same_ptr   => {
                            // We successfully unblocked our task pointer.
                            assert!(task_as_state == same_ptr);
                            let handle = BlockedTask::cast_from_uint(task_as_state);
                            // Because we are already awake, the handle we
                            // gave to this port shall already be empty.
                            handle.assert_already_awake();
                            false
                        }
                    }
                }
            }
        }
    }
}

impl<T> SelectPort<T> for PortOne<T> {
    fn recv_ready(self) -> Option<T> {
        let mut this = self;
        let packet = this.packet();

        // No further memory barrier is needed here to access the
        // payload. Some scenarios:
        //
        // 1) We encountered STATE_ONE above - the atomic_xchg was the acq barrier. We're fine.
        // 2) We encountered STATE_BOTH above and blocked. The sending task then ran us
        //    and ran on its thread. The sending task issued a read barrier when taking the
        //    pointer to the receiving task.
        // 3) We encountered STATE_BOTH above and blocked, but the receiving task (this task)
        //    is pinned to some other scheduler, so the sending task had to give us to
        //    a different scheduler for resuming. That send synchronized memory.
        unsafe {
            // See corresponding store() above in block_on for rationale.
            // FIXME(#8130) This can happen only in test builds.
            assert!((*packet).state.load(Acquire) == STATE_ONE);

            let payload = (*packet).payload.take();

            // The sender has closed up shop. Drop the packet.
            let _packet: ~Packet<T> = cast::transmute(this.void_packet);
            // Suppress the synchronizing actions in the finalizer. We're done with the packet.
            this.suppress_finalize = true;
            return payload;
        }
    }
}

impl<T> Peekable<T> for PortOne<T> {
    fn peek(&self) -> bool {
        unsafe {
            let packet: *mut Packet<T> = self.packet();
            let oldstate = (*packet).state.load(SeqCst);
            match oldstate {
                STATE_BOTH => false,
                STATE_ONE => (*packet).payload.is_some(),
                _ => rtabort!("peeked on a blocked task")
            }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for ChanOne<T> {
    fn drop(&self) {
        if self.suppress_finalize { return }

        unsafe {
            let this = cast::transmute_mut(self);
            let oldstate = (*this.packet()).state.swap(STATE_ONE, SeqCst);
            match oldstate {
                STATE_BOTH => {
                    // Port still active. It will destroy the Packet.
                },
                STATE_ONE => {
                    let _packet: ~Packet<T> = cast::transmute(this.void_packet);
                },
                task_as_state => {
                    // The port is blocked waiting for a message we will never send. Wake it.
                    assert!((*this.packet()).payload.is_none());
                    let recvr = BlockedTask::cast_from_uint(task_as_state);
                    do recvr.wake().map_consume |woken_task| {
                        let sched = Local::take::<Scheduler>();
                        sched.schedule_task(woken_task);
                    };
                }
            }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for PortOne<T> {
    fn drop(&self) {
        if self.suppress_finalize { return }

        unsafe {
            let this = cast::transmute_mut(self);
            let oldstate = (*this.packet()).state.swap(STATE_ONE, SeqCst);
            match oldstate {
                STATE_BOTH => {
                    // Chan still active. It will destroy the packet.
                },
                STATE_ONE => {
                    let _packet: ~Packet<T> = cast::transmute(this.void_packet);
                }
                task_as_state => {
                    // This case occurs during unwinding, when the blocked
                    // receiver was killed awake. The task can't still be
                    // blocked (we are it), but we need to free the handle.
                    let recvr = BlockedTask::cast_from_uint(task_as_state);
                    // FIXME(#7554)(bblum): Make this cfg(test) dependent.
                    // in a later commit.
                    assert!(recvr.wake().is_none());
                }
            }
        }
    }
}

struct StreamPayload<T> {
    val: T,
    next: PortOne<StreamPayload<T>>
}

type StreamChanOne<T> = ChanOne<StreamPayload<T>>;
type StreamPortOne<T> = PortOne<StreamPayload<T>>;

/// A channel with unbounded size.
pub struct Chan<T> {
    // FIXME #5372. Using Cell because we don't take &mut self
    next: Cell<StreamChanOne<T>>
}

/// An port with unbounded size.
pub struct Port<T> {
    // FIXME #5372. Using Cell because we don't take &mut self
    next: Cell<StreamPortOne<T>>
}

pub fn stream<T: Send>() -> (Port<T>, Chan<T>) {
    let (pone, cone) = oneshot();
    let port = Port { next: Cell::new(pone) };
    let chan = Chan { next: Cell::new(cone) };
    return (port, chan);
}

impl<T: Send> GenericChan<T> for Chan<T> {
    fn send(&self, val: T) {
        self.try_send(val);
    }
}

impl<T: Send> GenericSmartChan<T> for Chan<T> {
    fn try_send(&self, val: T) -> bool {
        let (next_pone, next_cone) = oneshot();
        let cone = self.next.take();
        self.next.put_back(next_cone);
        cone.try_send(StreamPayload { val: val, next: next_pone })
    }
}

impl<T> GenericPort<T> for Port<T> {
    fn recv(&self) -> T {
        match self.try_recv() {
            Some(val) => val,
            None => {
                fail!("receiving on closed channel");
            }
        }
    }

    fn try_recv(&self) -> Option<T> {
        let pone = self.next.take();
        match pone.try_recv() {
            Some(StreamPayload { val, next }) => {
                self.next.put_back(next);
                Some(val)
            }
            None => None
        }
    }
}

impl<T> Peekable<T> for Port<T> {
    fn peek(&self) -> bool {
        self.next.with_mut_ref(|p| p.peek())
    }
}

impl<T> Select for Port<T> {
    #[inline]
    fn optimistic_check(&mut self) -> bool {
        do self.next.with_mut_ref |pone| { pone.optimistic_check() }
    }

    #[inline]
    fn block_on(&mut self, sched: &mut Scheduler, task: BlockedTask) -> bool {
        let task = Cell::new(task);
        do self.next.with_mut_ref |pone| { pone.block_on(sched, task.take()) }
    }

    #[inline]
    fn unblock_from(&mut self) -> bool {
        do self.next.with_mut_ref |pone| { pone.unblock_from() }
    }
}

impl<T> SelectPort<(T, Port<T>)> for Port<T> {
    fn recv_ready(self) -> Option<(T, Port<T>)> {
        match self.next.take().recv_ready() {
            Some(StreamPayload { val, next }) => {
                self.next.put_back(next);
                Some((val, self))
            }
            None => None
        }
    }
}

pub struct SharedChan<T> {
    // Just like Chan, but a shared AtomicOption instead of Cell
    priv next: UnsafeAtomicRcBox<AtomicOption<StreamChanOne<T>>>
}

impl<T> SharedChan<T> {
    pub fn new(chan: Chan<T>) -> SharedChan<T> {
        let next = chan.next.take();
        let next = AtomicOption::new(~next);
        SharedChan { next: UnsafeAtomicRcBox::new(next) }
    }
}

impl<T: Send> GenericChan<T> for SharedChan<T> {
    fn send(&self, val: T) {
        self.try_send(val);
    }
}

impl<T: Send> GenericSmartChan<T> for SharedChan<T> {
    fn try_send(&self, val: T) -> bool {
        unsafe {
            let (next_pone, next_cone) = oneshot();
            let cone = (*self.next.get()).swap(~next_cone, SeqCst);
            cone.unwrap().try_send(StreamPayload { val: val, next: next_pone })
        }
    }
}

impl<T> Clone for SharedChan<T> {
    fn clone(&self) -> SharedChan<T> {
        SharedChan {
            next: self.next.clone()
        }
    }
}

pub struct SharedPort<T> {
    // The next port on which we will receive the next port on which we will receive T
    priv next_link: UnsafeAtomicRcBox<AtomicOption<PortOne<StreamPortOne<T>>>>
}

impl<T> SharedPort<T> {
    pub fn new(port: Port<T>) -> SharedPort<T> {
        // Put the data port into a new link pipe
        let next_data_port = port.next.take();
        let (next_link_port, next_link_chan) = oneshot();
        next_link_chan.send(next_data_port);
        let next_link = AtomicOption::new(~next_link_port);
        SharedPort { next_link: UnsafeAtomicRcBox::new(next_link) }
    }
}

impl<T: Send> GenericPort<T> for SharedPort<T> {
    fn recv(&self) -> T {
        match self.try_recv() {
            Some(val) => val,
            None => {
                fail!("receiving on a closed channel");
            }
        }
    }

    fn try_recv(&self) -> Option<T> {
        unsafe {
            let (next_link_port, next_link_chan) = oneshot();
            let link_port = (*self.next_link.get()).swap(~next_link_port, SeqCst);
            let link_port = link_port.unwrap();
            let data_port = link_port.recv();
            let (next_data_port, res) = match data_port.try_recv() {
                Some(StreamPayload { val, next }) => {
                    (next, Some(val))
                }
                None => {
                    let (next_data_port, _) = oneshot();
                    (next_data_port, None)
                }
            };
            next_link_chan.send(next_data_port);
            return res;
        }
    }
}

impl<T> Clone for SharedPort<T> {
    fn clone(&self) -> SharedPort<T> {
        SharedPort {
            next_link: self.next_link.clone()
        }
    }
}

// XXX: Need better name
type MegaPipe<T> = (SharedPort<T>, SharedChan<T>);

pub fn megapipe<T: Send>() -> MegaPipe<T> {
    let (port, chan) = stream();
    (SharedPort::new(port), SharedChan::new(chan))
}

impl<T: Send> GenericChan<T> for MegaPipe<T> {
    fn send(&self, val: T) {
        match *self {
            (_, ref c) => c.send(val)
        }
    }
}

impl<T: Send> GenericSmartChan<T> for MegaPipe<T> {
    fn try_send(&self, val: T) -> bool {
        match *self {
            (_, ref c) => c.try_send(val)
        }
    }
}

impl<T: Send> GenericPort<T> for MegaPipe<T> {
    fn recv(&self) -> T {
        match *self {
            (ref p, _) => p.recv()
        }
    }

    fn try_recv(&self) -> Option<T> {
        match *self {
            (ref p, _) => p.try_recv()
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use option::*;
    use rt::test::*;
    use cell::Cell;
    use iter::Times;

    #[test]
    fn oneshot_single_thread_close_port_first() {
        // Simple test of closing without sending
        do run_in_newsched_task {
            let (port, _chan) = oneshot::<int>();
            { let _p = port; }
        }
    }

    #[test]
    fn oneshot_single_thread_close_chan_first() {
        // Simple test of closing without sending
        do run_in_newsched_task {
            let (_port, chan) = oneshot::<int>();
            { let _c = chan; }
        }
    }

    #[test]
    fn oneshot_single_thread_send_port_close() {
        // Testing that the sender cleans up the payload if receiver is closed
        do run_in_newsched_task {
            let (port, chan) = oneshot::<~int>();
            { let _p = port; }
            chan.send(~0);
        }
    }

    #[test]
    fn oneshot_single_thread_recv_chan_close() {
        // Receiving on a closed chan will fail
        do run_in_newsched_task {
            let res = do spawntask_try {
                let (port, chan) = oneshot::<~int>();
                { let _c = chan; }
                port.recv();
            };
            // What is our res?
            rtdebug!("res is: %?", res.is_err());
            assert!(res.is_err());
        }
    }

    #[test]
    fn oneshot_single_thread_send_then_recv() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<~int>();
            chan.send(~10);
            assert!(port.recv() == ~10);
        }
    }

    #[test]
    fn oneshot_single_thread_try_send_open() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<int>();
            assert!(chan.try_send(10));
            assert!(port.recv() == 10);
        }
    }

    #[test]
    fn oneshot_single_thread_try_send_closed() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<int>();
            { let _p = port; }
            assert!(!chan.try_send(10));
        }
    }

    #[test]
    fn oneshot_single_thread_try_recv_open() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<int>();
            chan.send(10);
            assert!(port.try_recv() == Some(10));
        }
    }

    #[test]
    fn oneshot_single_thread_try_recv_closed() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<int>();
            { let _c = chan; }
            assert!(port.try_recv() == None);
        }
    }

    #[test]
    fn oneshot_single_thread_peek_data() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<int>();
            assert!(!port.peek());
            chan.send(10);
            assert!(port.peek());
        }
    }

    #[test]
    fn oneshot_single_thread_peek_close() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<int>();
            { let _c = chan; }
            assert!(!port.peek());
            assert!(!port.peek());
        }
    }

    #[test]
    fn oneshot_single_thread_peek_open() {
        do run_in_newsched_task {
            let (port, _) = oneshot::<int>();
            assert!(!port.peek());
        }
    }

    #[test]
    fn oneshot_multi_task_recv_then_send() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<~int>();
            let port_cell = Cell::new(port);
            do spawntask_immediately {
                assert!(port_cell.take().recv() == ~10);
            }

            chan.send(~10);
        }
    }

    #[test]
    fn oneshot_multi_task_recv_then_close() {
        do run_in_newsched_task {
            let (port, chan) = oneshot::<~int>();
            let port_cell = Cell::new(port);
            let chan_cell = Cell::new(chan);
            do spawntask_later {
                let _cell = chan_cell.take();
            }
            let res = do spawntask_try {
                assert!(port_cell.take().recv() == ~10);
            };
            assert!(res.is_err());
        }
    }

    #[test]
    fn oneshot_multi_thread_close_stress() {
        for stress_factor().times {
            do run_in_newsched_task {
                let (port, chan) = oneshot::<int>();
                let port_cell = Cell::new(port);
                let thread = do spawntask_thread {
                    let _p = port_cell.take();
                };
                let _chan = chan;
                thread.join();
            }
        }
    }

    #[test]
    fn oneshot_multi_thread_send_close_stress() {
        for stress_factor().times {
            do run_in_newsched_task {
                let (port, chan) = oneshot::<int>();
                let chan_cell = Cell::new(chan);
                let port_cell = Cell::new(port);
                let thread1 = do spawntask_thread {
                    let _p = port_cell.take();
                };
                let thread2 = do spawntask_thread {
                    let c = chan_cell.take();
                    c.send(1);
                };
                thread1.join();
                thread2.join();
            }
        }
    }

    #[test]
    fn oneshot_multi_thread_recv_close_stress() {
        for stress_factor().times {
            do run_in_newsched_task {
                let (port, chan) = oneshot::<int>();
                let chan_cell = Cell::new(chan);
                let port_cell = Cell::new(port);
                let thread1 = do spawntask_thread {
                    let port_cell = Cell::new(port_cell.take());
                    let res = do spawntask_try {
                        port_cell.take().recv();
                    };
                    assert!(res.is_err());
                };
                let thread2 = do spawntask_thread {
                    let chan_cell = Cell::new(chan_cell.take());
                    do spawntask {
                        chan_cell.take();
                    }
                };
                thread1.join();
                thread2.join();
            }
        }
    }

    #[test]
    fn oneshot_multi_thread_send_recv_stress() {
        for stress_factor().times {
            do run_in_newsched_task {
                let (port, chan) = oneshot::<~int>();
                let chan_cell = Cell::new(chan);
                let port_cell = Cell::new(port);
                let thread1 = do spawntask_thread {
                    chan_cell.take().send(~10);
                };
                let thread2 = do spawntask_thread {
                    assert!(port_cell.take().recv() == ~10);
                };
                thread1.join();
                thread2.join();
            }
        }
    }

    #[test]
    fn stream_send_recv_stress() {
        for stress_factor().times {
            do run_in_mt_newsched_task {
                let (port, chan) = stream::<~int>();

                send(chan, 0);
                recv(port, 0);

                fn send(chan: Chan<~int>, i: int) {
                    if i == 10 { return }

                    let chan_cell = Cell::new(chan);
                    do spawntask_random {
                        let chan = chan_cell.take();
                        chan.send(~i);
                        send(chan, i + 1);
                    }
                }

                fn recv(port: Port<~int>, i: int) {
                    if i == 10 { return }

                    let port_cell = Cell::new(port);
                    do spawntask_random {
                        let port = port_cell.take();
                        assert!(port.recv() == ~i);
                        recv(port, i + 1);
                    };
                }
            }
        }
    }

    #[test]
    fn recv_a_lot() {
        // Regression test that we don't run out of stack in scheduler context
        do run_in_newsched_task {
            let (port, chan) = stream();
            for 10000.times { chan.send(()) }
            for 10000.times { port.recv() }
        }
    }

    #[test]
    fn shared_chan_stress() {
        do run_in_mt_newsched_task {
            let (port, chan) = stream();
            let chan = SharedChan::new(chan);
            let total = stress_factor() + 100;
            for total.times {
                let chan_clone = chan.clone();
                do spawntask_random {
                    chan_clone.send(());
                }
            }

            for total.times {
                port.recv();
            }
        }
    }

    #[test]
    fn shared_port_stress() {
        do run_in_mt_newsched_task {
            // XXX: Removing these type annotations causes an ICE
            let (end_port, end_chan) = stream::<()>();
            let (port, chan) = stream::<()>();
            let end_chan = SharedChan::new(end_chan);
            let port = SharedPort::new(port);
            let total = stress_factor() + 100;
            for total.times {
                let end_chan_clone = end_chan.clone();
                let port_clone = port.clone();
                do spawntask_random {
                    port_clone.recv();
                    end_chan_clone.send(());
                }
            }

            for total.times {
                chan.send(());
            }

            for total.times {
                end_port.recv();
            }
        }
    }

    #[test]
    fn shared_port_close_simple() {
        do run_in_mt_newsched_task {
            let (port, chan) = stream::<()>();
            let port = SharedPort::new(port);
            { let _chan = chan; }
            assert!(port.try_recv().is_none());
        }
    }

    #[test]
    fn shared_port_close() {
        do run_in_mt_newsched_task {
            let (end_port, end_chan) = stream::<bool>();
            let (port, chan) = stream::<()>();
            let end_chan = SharedChan::new(end_chan);
            let port = SharedPort::new(port);
            let chan = SharedChan::new(chan);
            let send_total = 10;
            let recv_total = 20;
            do spawntask_random {
                for send_total.times {
                    let chan_clone = chan.clone();
                    do spawntask_random {
                        chan_clone.send(());
                    }
                }
            }
            let end_chan_clone = end_chan.clone();
            do spawntask_random {
                for recv_total.times {
                    let port_clone = port.clone();
                    let end_chan_clone = end_chan_clone.clone();
                    do spawntask_random {
                        let recvd = port_clone.try_recv().is_some();
                        end_chan_clone.send(recvd);
                    }
                }
            }

            let mut recvd = 0;
            for recv_total.times {
                recvd += if end_port.recv() { 1 } else { 0 };
            }

            assert!(recvd == send_total);
        }
    }

    #[test]
    fn megapipe_stress() {
        use rand;
        use rand::RngUtil;

        do run_in_mt_newsched_task {
            let (end_port, end_chan) = stream::<()>();
            let end_chan = SharedChan::new(end_chan);
            let pipe = megapipe();
            let total = stress_factor() + 10;
            let mut rng = rand::rng();
            for total.times {
                let msgs = rng.gen_uint_range(0, 10);
                let pipe_clone = pipe.clone();
                let end_chan_clone = end_chan.clone();
                do spawntask_random {
                    for msgs.times {
                        pipe_clone.send(());
                    }
                    for msgs.times {
                        pipe_clone.recv();
                    }
                }

                end_chan_clone.send(());
            }

            for total.times {
                end_port.recv();
            }
        }
    }

}
