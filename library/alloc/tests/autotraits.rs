fn require_sync<T: Sync>(_: T) {}
fn require_send_sync<T: Send + Sync>(_: T) {}

struct NotSend(#[allow(dead_code)] *const ());
unsafe impl Sync for NotSend {}

#[test]
fn test_btree_map() {
    // Tests of this form are prone to https://github.com/rust-lang/rust/issues/64552.
    //
    // In theory the async block's future would be Send if the value we hold
    // across the await point is Send, and Sync if the value we hold across the
    // await point is Sync.
    //
    // We test autotraits in this convoluted way, instead of a straightforward
    // `require_send_sync::<TypeIWantToTest>()`, because the interaction with
    // coroutines exposes some current limitations in rustc's ability to prove a
    // lifetime bound on the erased coroutine witness types. See the above link.
    //
    // A typical way this would surface in real code is:
    //
    //     fn spawn<T: Future + Send>(_: T) {}
    //
    //     async fn f() {
    //         let map = BTreeMap::<u32, Box<dyn Send + Sync>>::new();
    //         for _ in &map {
    //             async {}.await;
    //         }
    //     }
    //
    //     fn main() {
    //         spawn(f());
    //     }
    //
    // where with some unintentionally overconstrained Send impls in alloc's
    // internals, the future might incorrectly not be Send even though every
    // single type involved in the program is Send and Sync.
    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::Iter<'_, &u32, &u32>>;
        async {}.await;
    });

    // Testing like this would not catch all issues that the above form catches.
    require_send_sync(None::<alloc::collections::btree_map::Iter<'_, &u32, &u32>>);

    require_sync(async {
        let _v = None::<alloc::collections::btree_map::Iter<'_, u32, NotSend>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::BTreeMap<&u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<
            alloc::collections::btree_map::ExtractIf<'_, &u32, &u32, fn(&&u32, &mut &u32) -> bool>,
        >;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::Entry<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::IntoIter<&u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::IntoKeys<&u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::IntoValues<&u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::Iter<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::IterMut<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::Keys<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::OccupiedEntry<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::OccupiedError<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::Range<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::RangeMut<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::VacantEntry<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::Values<'_, &u32, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_map::ValuesMut<'_, &u32, &u32>>;
        async {}.await;
    });
}

#[test]
fn test_btree_set() {
    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::BTreeSet<&u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::Difference<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::ExtractIf<'_, &u32, fn(&&u32) -> bool>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::Intersection<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::IntoIter<&u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::Iter<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::Range<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::SymmetricDifference<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::btree_set::Union<'_, &u32>>;
        async {}.await;
    });
}

#[test]
fn test_binary_heap() {
    require_send_sync(async {
        let _v = None::<alloc::collections::binary_heap::BinaryHeap<&u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::binary_heap::Drain<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::binary_heap::DrainSorted<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::binary_heap::IntoIter<&u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::binary_heap::IntoIterSorted<&u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::binary_heap::Iter<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::binary_heap::PeekMut<'_, &u32>>;
        async {}.await;
    });
}

#[test]
fn test_linked_list() {
    require_send_sync(async {
        let _v = None::<alloc::collections::linked_list::Cursor<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::linked_list::CursorMut<'_, &u32>>;
        async {}.await;
    });

    // FIXME
    /*
    require_send_sync(async {
        let _v =
            None::<alloc::collections::linked_list::ExtractIf<'_, &u32, fn(&mut &u32) -> bool>>;
        async {}.await;
    });
    */

    require_send_sync(async {
        let _v = None::<alloc::collections::linked_list::IntoIter<&u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::linked_list::Iter<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::linked_list::IterMut<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::linked_list::LinkedList<&u32>>;
        async {}.await;
    });
}

#[test]
fn test_vec_deque() {
    require_send_sync(async {
        let _v = None::<alloc::collections::vec_deque::Drain<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::vec_deque::IntoIter<&u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::vec_deque::Iter<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::vec_deque::IterMut<'_, &u32>>;
        async {}.await;
    });

    require_send_sync(async {
        let _v = None::<alloc::collections::vec_deque::VecDeque<&u32>>;
        async {}.await;
    });
}
