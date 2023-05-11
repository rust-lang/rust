incremental_unrecognized_depnode = unrecognized `DepNode` variant: {$name}

incremental_missing_depnode = missing `DepNode` variant

incremental_missing_if_this_changed = no `#[rustc_if_this_changed]` annotation detected

incremental_no_path = no path from `{$source}` to `{$target}`

incremental_ok = OK

incremental_unknown_reuse_kind = unknown cgu-reuse-kind `{$kind}` specified

incremental_missing_query_depgraph =
    found CGU-reuse attribute but `-Zquery-dep-graph` was not specified

incremental_malformed_cgu_name =
    found malformed codegen unit name `{$user_path}`. codegen units names must always start with the name of the crate (`{$crate_name}` in this case).

incremental_no_module_named =
    no module named `{$user_path}` (mangled: {$cgu_name}). available modules: {$cgu_names}

incremental_field_associated_value_expected = associated value expected for `{$name}`

incremental_no_field = no field `{$name}`

incremental_assertion_auto =
    `except` specified DepNodes that can not be affected for "{$name}": "{$e}"

incremental_undefined_clean_dirty_assertions_item =
    clean/dirty auto-assertions not yet defined for Node::Item.node={$kind}

incremental_undefined_clean_dirty_assertions =
    clean/dirty auto-assertions not yet defined for {$kind}

incremental_repeated_depnode_label = dep-node label `{$label}` is repeated

incremental_unrecognized_depnode_label = dep-node label `{$label}` not recognized

incremental_not_dirty = `{$dep_node_str}` should be dirty but is not

incremental_not_clean = `{$dep_node_str}` should be clean but is not

incremental_not_loaded = `{$dep_node_str}` should have been loaded from disk but it was not

incremental_unknown_item = unknown item `{$name}`

incremental_no_cfg = no cfg attribute

incremental_associated_value_expected_for = associated value expected for `{$ident}`

incremental_associated_value_expected = expected an associated value

incremental_unchecked_clean = found unchecked `#[rustc_clean]` attribute

incremental_delete_old = unable to delete old {$name} at `{$path}`: {$err}

incremental_create_new = failed to create {$name} at `{$path}`: {$err}

incremental_write_new = failed to write {$name} to `{$path}`: {$err}

incremental_canonicalize_path = incremental compilation: error canonicalizing path `{$path}`: {$err}

incremental_create_incr_comp_dir =
    could not create incremental compilation {$tag} directory `{$path}`: {$err}

incremental_create_lock =
    incremental compilation: could not create session directory lock file: {$lock_err}
incremental_lock_unsupported =
    the filesystem for the incremental path at {$session_dir} does not appear to support locking, consider changing the incremental path to a filesystem that supports locking or disable incremental compilation
incremental_cargo_help_1 =
    incremental compilation can be disabled by setting the environment variable CARGO_INCREMENTAL=0 (see https://doc.rust-lang.org/cargo/reference/profiles.html#incremental)
incremental_cargo_help_2 =
    the entire build directory can be changed to a different filesystem by setting the environment variable CARGO_TARGET_DIR to a different path (see https://doc.rust-lang.org/cargo/reference/config.html#buildtarget-dir)

incremental_delete_lock =
    error deleting lock file for incremental compilation session directory `{$path}`: {$err}

incremental_hard_link_failed =
    hard linking files in the incremental compilation cache failed. copying files instead. consider moving the cache directory to a file system which supports hard linking in session dir `{$path}`

incremental_delete_partial = failed to delete partly initialized session dir `{$path}`: {$err}

incremental_delete_full = error deleting incremental compilation session directory `{$path}`: {$err}

incremental_finalize = error finalizing incremental compilation session directory `{$path}`: {$err}

incremental_invalid_gc_failed =
    failed to garbage collect invalid incremental compilation session directory `{$path}`: {$err}

incremental_finalized_gc_failed =
    failed to garbage collect finalized incremental compilation session directory `{$path}`: {$err}

incremental_session_gc_failed =
    failed to garbage collect incremental compilation session directory `{$path}`: {$err}

incremental_assert_not_loaded =
    we asserted that the incremental cache should not be loaded, but it was loaded

incremental_assert_loaded =
    we asserted that an existing incremental cache directory should be successfully loaded, but it was not

incremental_delete_incompatible =
    failed to delete invalidated or incompatible incremental compilation session directory contents `{$path}`: {$err}

incremental_load_dep_graph = could not load dep-graph from `{$path}`: {$err}

incremental_decode_incr_cache = could not decode incremental cache: {$err}

incremental_write_dep_graph = failed to write dependency graph to `{$path}`: {$err}

incremental_move_dep_graph = failed to move dependency graph from `{$from}` to `{$to}`: {$err}

incremental_create_dep_graph = failed to create dependency graph at `{$path}`: {$err}

incremental_copy_workproduct_to_cache =
    error copying object file `{$from}` to incremental directory as `{$to}`: {$err}

incremental_delete_workproduct = file-system error deleting outdated file `{$path}`: {$err}
