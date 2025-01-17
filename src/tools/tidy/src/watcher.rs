//! Checks that text between tags unchanged, emitting warning otherwise,
//! allowing asserting that code in different places over codebase is in sync.
//!
//! This works via hashing text between tags and saving hash in tidy.
//!
//! Usage:
//!
//! some.rs:
//! // tidy-ticket-foo
//! const FOO: usize = 42;
//! // tidy-ticket-foo
//!
//! some.sh:
//! # tidy-ticket-foo
//! export FOO=42
//! # tidy-ticket-foo
use std::fs;
use std::path::Path;

use md5::{Digest, Md5};

#[cfg(test)]
mod tests;

/// Return hash for source text between 2 tag occurrence,
/// ignoring lines where tag written
///
/// Expecting:
/// tag is not multiline
/// source always have at least 2 occurrence of tag (>2 ignored)
fn span_hash(source: &str, tag: &str, bad: &mut bool) -> Result<String, ()> {
    let start_idx = match source.find(tag) {
        Some(idx) => idx,
        None => return Err(tidy_error!(bad, "tag {} should exist in provided text", tag)),
    };
    let end_idx = {
        let end = match source[start_idx + tag.len()..].find(tag) {
            // index from source start
            Some(idx) => start_idx + tag.len() + idx,
            None => return Err(tidy_error!(bad, "tag end {} should exist in provided text", tag)),
        };
        // second line with tag can contain some other text before tag, ignore it
        // by finding position of previous line ending
        //
        // FIXME: what if line ending is \r\n? In that case \r will be hashed too
        let offset = source[start_idx..end].rfind('\n').unwrap();
        start_idx + offset
    };

    let mut hasher = Md5::new();

    source[start_idx..end_idx]
        .lines()
        // skip first line with tag
        .skip(1)
        // hash next lines, ignoring end trailing whitespaces
        .for_each(|line| {
            let trimmed = line.trim_end();
            hasher.update(trimmed);
        });
    Ok(format!("{:x}", hasher.finalize()))
}

fn check_entry(entry: &ListEntry<'_>, group_idx: usize, bad: &mut bool, root_path: &Path) {
    let file = fs::read_to_string(root_path.join(Path::new(entry.0)))
        .unwrap_or_else(|e| panic!("{:?}, path: {}", e, entry.0));
    let actual_hash = span_hash(&file, entry.2, bad).unwrap();
    if actual_hash != entry.1 {
        // Write tidy error description for wather only once.
        // Will not work if there was previous errors of other types.
        if *bad == false {
            tidy_error!(
                bad,
                "The code blocks tagged with tidy watcher has changed.\n\
                 It's likely that code blocks with the following tags need to be changed too. Check src/tools/tidy/src/watcher.rs, find tag/hash in TIDY_WATCH_LIST list \
                and verify that sources for provided group of tags in sync. Once that done, run tidy again and update hashes in TIDY_WATCH_LIST with provided actual hashes."
            )
        }
        tidy_error!(
            bad,
            "hash for tag `{}` in path `{}` mismatch:\n  actual: `{}`, expected: `{}`\n  \
            Verify that tags `{:?}` in sync.",
            entry.2,
            entry.0,
            actual_hash,
            entry.1,
            TIDY_WATCH_LIST[group_idx].iter().map(|e| e.2).collect::<Vec<&str>>()
        );
    }
}

macro_rules! add_group {
    ($($entry:expr),*) => {
        &[$($entry),*]
    };
}

/// (path, hash, tag)
type ListEntry<'a> = (&'a str, &'a str, &'a str);

/// List of tags to watch, along with paths and hashes
#[rustfmt::skip]
const TIDY_WATCH_LIST: &[&[ListEntry<'_>]] = &[
    add_group!(
        ("compiler/rustc_ast/src/token.rs", "2df3e863dc4caffb31a7ddf001517232", "tidy-ticket-ast-from_token"),
        ("compiler/rustc_ast/src/token.rs", "3279df5da05e0455f45630917fe2a797", "tidy-ticket-ast-can_begin_literal_maybe_minus"),
        ("compiler/rustc_parse/src/parser/expr.rs", "479655a8587512fc26f7361d7bbd75a5", "tidy-ticket-rustc_parse-can_begin_literal_maybe_minus")
    ),

    add_group!(
        ("compiler/rustc_builtin_macros/src/assert/context.rs", "dbac73cc47a451d4822ccd3010c40a0f", "tidy-ticket-all-expr-kinds"),
        ("tests/ui/macros/rfc-2011-nicer-assert-messages/all-expr-kinds.rs", "78ce54cc25baeac3ae07c876db25180c", "tidy-ticket-all-expr-kinds")
    ),

    add_group!(
        ("compiler/rustc_const_eval/src/interpret/validity.rs", "c4e96ecd3f81dcb54541b8ea41042e8f", "tidy-ticket-try_visit_primitive"),
        ("compiler/rustc_const_eval/src/interpret/validity.rs", "cbe69005510c1a87ab07db601c0d36b8", "tidy-ticket-visit_value")
    ),

    // sync self-profile-events help mesage with actual list of events
    add_group!(
        ("compiler/rustc_data_structures/src/profiling.rs", "881e7899c7d6904af1bc000594ee0418", "tidy-ticket-self-profile-events"),
        ("compiler/rustc_session/src/options.rs", "012ee5a3b61ee1377744e5c6913fa00a", "tidy-ticket-self-profile-events")
    ),

    add_group!(
        ("compiler/rustc_errors/src/json.rs", "3963a8c4eee7f87eeb076622b8a92891", "tidy-ticket-UnusedExterns"),
        ("src/librustdoc/doctest.rs", "14da85663568149c9b21686f4b7fa7b0", "tidy-ticket-UnusedExterns")
    ),

    add_group!(
        ("compiler/rustc_middle/src/ty/util.rs", "cae64b1bc854e7ee81894212facb5bfa", "tidy-ticket-static_ptr_ty"),
        ("compiler/rustc_middle/src/ty/util.rs", "6f5ead08474b4d3e358db5d3c7aef970", "tidy-ticket-thread_local_ptr_ty")
    ),

    // desynced, pieces in compiler/rustc_pattern_analysis/src/rustc.rs
    // add_group!(
    //     ("compiler/rustc_pattern_analysis/src/constructor.rs", "c17706947fc814aa5648972a5b3dc143", "tidy-ticket-arity"),
    //     // ("compiler/rustc_mir_build/src/thir/pattern/deconstruct_pat.rs", "7ce77b84c142c22530b047703ef209f0", "tidy-ticket-wildcards")
    // ),

    add_group!(
        ("compiler/rustc_monomorphize/src/partitioning.rs", "f4f33e9c14f4e0c3a20b5240ae36a7c8", "tidy-ticket-short_description"),
        ("compiler/rustc_codegen_ssa/src/back/write.rs", "5286f7f76fcf564c98d7a8eaeec39b18", "tidy-ticket-short_description")
    ),

    add_group!(
        ("compiler/rustc_session/src/config/sigpipe.rs", "330e0776ba5a6c0a7439a5235297f08f", "tidy-ticket-sigpipe"),
        ("library/std/src/sys/pal/unix/mod.rs", "2cdc37081831cdcf44f3331efbe440af", "tidy-ticket-sigpipe")
    ),

    add_group!(
        ("compiler/rustc_next_trait_solver/src/solve/assembly/structural_traits.rs", "8726918d084e0ac5bb07184008403f88", "tidy-ticket-extract_tupled_inputs_and_output_from_callable"),
        ("compiler/rustc_trait_selection/src/traits/select/candidate_assembly.rs", "be2967d323633c7458533c6cec228273", "tidy-ticket-assemble_fn_pointer_candidates")
    ),

    add_group!(
        ("compiler/rustc_trait_selection/src/traits/project.rs", "262d10feb1b7ba8d3ffb4a95314bf404", "tidy-ticket-assemble_candidates_from_impls-UserDefined"),
        ("compiler/rustc_ty_utils/src/instance.rs", "a51a6022efb405c5ee86acdf49ec222d", "tidy-ticket-resolve_associated_item-UserDefined")
    ),

    // desynced, pieces in compiler/rustc_hir_analysis/src/lib.rs missing?
    //add_group!( // bad
    //    ("compiler/rustc_hir_analysis/src/lib.rs", "842e23fb65caf3a96681686131093316", "tidy-ticket-sess-time-item_types_checking"),
    //    ("src/librustdoc/core.rs", "85d9dd0cbb94fd521e2d15a8ed38a75f", "tidy-ticket-sess-time-item_types_checking")
    // ),

    add_group!(
        ("library/core/src/ptr/metadata.rs", "357c958a096c33bed67cfc7212d940a2", "tidy-ticket-static_assert_expected_bounds_for_metadata"),
        ("library/core/tests/ptr.rs", "d8c47e54b871d72dfdce56e6a89b5c31", "tidy-ticket-static_assert_expected_bounds_for_metadata")
    ),
];

pub fn check(root_path: &Path, bad: &mut bool) {
    for (group_idx, group) in TIDY_WATCH_LIST.iter().enumerate() {
        for entry in group.iter() {
            check_entry(entry, group_idx, bad, root_path);
        }
    }
}
