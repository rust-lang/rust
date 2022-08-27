query_system_reentrant = internal compiler error: re-entrant incremental verify failure, suppressing message

query_system_increment_compilation = internal compiler error: encountered incremental compilation error with {$dep_node}
    .help = This is a known issue with the compiler. Run {$run_cmd} to allow your project to compile

query_system_increment_compilation_note1 = Please follow the instructions below to create a bug report with the provided information
query_system_increment_compilation_note2 = See <https://github.com/rust-lang/rust/issues/84970> for more information

query_system_cycle = cycle detected when {$stack_bottom}

query_system_cycle_usage = cycle used when {$usage}

query_system_cycle_stack_single = ...which immediately requires {$stack_bottom} again

query_system_cycle_stack_multiple = ...which again requires {$stack_bottom}, completing the cycle

query_system_cycle_recursive_ty_alias = type aliases cannot be recursive
query_system_cycle_recursive_ty_alias_help1 = consider using a struct, enum, or union instead to break the cycle
query_system_cycle_recursive_ty_alias_help2 = see <https://doc.rust-lang.org/reference/types.html#recursive-types> for more information

query_system_cycle_recursive_trait_alias = trait aliases cannot be recursive

query_system_cycle_which_requires = ...which requires {$desc}...

query_system_query_overflow = queries overflow the depth limit!
