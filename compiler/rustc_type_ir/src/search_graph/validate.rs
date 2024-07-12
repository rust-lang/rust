use super::*;

impl<D: Delegate<Cx = X>, X: Cx> SearchGraph<D> {
    #[allow(rustc::potential_query_instability)]
    pub(super) fn check_invariants(&self) {
        if !cfg!(debug_assertions) {
            return;
        }

        let SearchGraph { mode: _, stack, provisional_cache, _marker } = self;
        if stack.is_empty() {
            assert!(provisional_cache.is_empty());
        }

        for (depth, entry) in stack.iter_enumerated() {
            let StackEntry {
                input,
                available_depth: _,
                reached_depth: _,
                non_root_cycle_participant,
                encountered_overflow: _,
                has_been_used,
                ref nested_goals,
                provisional_result,
            } = *entry;
            let cache_entry = provisional_cache.get(&entry.input).unwrap();
            assert_eq!(cache_entry.stack_depth, Some(depth));
            if let Some(head) = non_root_cycle_participant {
                assert!(head < depth);
                assert!(nested_goals.is_empty());
                assert_ne!(stack[head].has_been_used, None);

                let mut current_root = head;
                while let Some(parent) = stack[current_root].non_root_cycle_participant {
                    current_root = parent;
                }
                assert!(stack[current_root].nested_goals.contains(&input));
            }

            if !nested_goals.is_empty() {
                assert!(provisional_result.is_some() || has_been_used.is_some());
                for entry in stack.iter().take(depth.as_usize()) {
                    assert_eq!(nested_goals.get(&entry.input), None);
                }
            }
        }

        for (&input, entry) in &self.provisional_cache {
            let ProvisionalCacheEntry { stack_depth, with_coinductive_stack, with_inductive_stack } =
                entry;
            assert!(
                stack_depth.is_some()
                    || with_coinductive_stack.is_some()
                    || with_inductive_stack.is_some()
            );

            if let &Some(stack_depth) = stack_depth {
                assert_eq!(stack[stack_depth].input, input);
            }

            let check_detached = |detached_entry: &DetachedEntry<X>| {
                let DetachedEntry { head, result: _ } = *detached_entry;
                assert_ne!(stack[head].has_been_used, None);
            };

            if let Some(with_coinductive_stack) = with_coinductive_stack {
                check_detached(with_coinductive_stack);
            }

            if let Some(with_inductive_stack) = with_inductive_stack {
                check_detached(with_inductive_stack);
            }
        }
    }
}
