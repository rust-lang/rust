//@compile-flags: -Zmiri-preemption-rate=0 -Zmiri-compare-exchange-weak-failure-rate=0
use std::sync::mpsc::channel;
use std::thread;

/// This test aims to trigger a race condition that causes a double free in the unbounded channel
/// implementation. The test relies on a particular thread scheduling to happen as annotated by the
/// comments below.
fn main() {
    let (s1, r) = channel::<u64>();
    let s2 = s1.clone();

    let t1 = thread::spawn(move || {
        // 1. The first action executed is an attempt to send the first value in the channel. This
        //    will begin to initialize the channel but will stop at a critical momement as
        //    indicated by the `yield_now()` call in the `start_send` method of the implementation.
        let _ = s1.send(42);
        // 4. The sender is re-scheduled and it finishes the initialization of the channel by
        //    setting head.block to the same value as tail.block. It then proceeds to publish its
        //    value but observes that the channel has already disconnected (due to the concurrent
        //    call of `discard_all_messages`) and aborts the send.
    });
    std::thread::yield_now();

    // 2. A second sender attempts to send a value while the channel is in a half-initialized
    //    state. Here, half-initialized means that the `tail.block` pointer points to a valid block
    //    but `head.block` is still null. This condition is ensured by the yield of step 1. When
    //    this call returns the channel state has tail.index != head.index, tail.block != NULL, and
    //    head.block = NULL.
    s2.send(42).unwrap();
    // 3. This thread continues with dropping the one and only receiver. When all receivers are
    //    gone `discard_all_messages` will attempt to drop all currently sent values and
    //    de-allocate all the blocks. If `tail.block != NULL` but `head.block = NULL` the
    //    implementation waits for the initializing sender to finish by spinning/yielding.
    drop(r);
    // 5. This thread is rescheduled and `discard_all_messages` observes the head.block pointer set
    //    by step 4 and proceeds with deallocation. In the problematic version of the code
    //    `head.block` is simply read via an `Acquire` load and not swapped with NULL. After this
    //    call returns the channel state has tail.index = head.index, tail.block = NULL, and
    //    head.block != NULL.
    t1.join().unwrap();
    // 6. The last sender (s2) is dropped here which also attempts to cleanup any data in the
    //    channel. It observes `tail.index = head.index` and so it doesn't attempt to cleanup any
    //    messages but it also observes that `head.block != NULL` and attempts to deallocate it.
    //    This is however already deallocated by `discard_all_messages`, leading to a double free.
}
