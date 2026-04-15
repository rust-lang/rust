//! Compile-time and runtime checks for kernel boundary contract types.
//!
//! These tests ensure kernel-facing bridge functions keep returning canonical
//! ThingOS boundary types and that KindId constants remain aligned with
//! kindc-generated schema output.
//!
//! # Guardrail coverage
//!
//! | Surface                              | Test                                    |
//! |--------------------------------------|-----------------------------------------|
//! | task bridge signatures               | [`bridge_signatures_return_canonical_types`] |
//! | job bridge signatures                | [`bridge_signatures_return_canonical_types`] |
//! | group bridge signatures              | [`bridge_signatures_return_canonical_types`] |
//! | authority bridge signatures          | [`bridge_signatures_return_canonical_types`] |
//! | place bridge signatures              | [`bridge_signatures_return_canonical_types`] |
//! | message bridge signatures            | [`bridge_signatures_return_canonical_types`] |
//! | KIND_ID_THINGOS_MESSAGE              | [`kind_ids_match_kindc_generated_constants`] |
//! | KIND_ID_THINGOS_JOB_EXIT             | [`kind_ids_match_kindc_generated_constants`] |
//! | KIND_ID_THINGOS_AUTHORITY            | [`kind_ids_match_kindc_generated_constants`] |
//! | KIND_ID_THINGOS_TASK / TASK_STATE    | [`kind_ids_match_kindc_generated_constants`] |
//! | KIND_ID_THINGOS_JOB / JOB_STATE / JOB_WAIT_RESULT | [`kind_ids_match_kindc_generated_constants`] |
//! | KIND_ID_THINGOS_GROUP                | [`kind_ids_match_kindc_generated_constants`] |
//! | KIND_ID_THINGOS_PLACE                | [`kind_ids_match_kindc_generated_constants`] |

use thingos::message::KindId;

fn parse_generated_kind_id(name: &str) -> [u8; 16] {
    let src = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/kindc/fixtures/generated/mod.rs"
    ));
    let needle = alloc::format!("pub const {}: [u8; 16] = [", name);
    let start = src.find(&needle).expect("kind id constant must exist") + needle.len();
    let tail = &src[start..];
    let end = tail.find("];\n").expect("kind id constant must terminate with ];");
    let body = &tail[..end];

    let mut out = [0u8; 16];
    let mut count = 0usize;
    for tok in body.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        assert!(count < 16, "kind id has too many bytes");
        let hex = tok.strip_prefix("0x").expect("kind id bytes must be hex");
        out[count] = u8::from_str_radix(hex, 16).expect("invalid kind id byte");
        count += 1;
    }
    assert_eq!(count, 16, "kind id must contain exactly 16 bytes");
    out
}

#[test]
fn bridge_signatures_return_canonical_types() {
    let _: fn(crate::task::ThreadState) -> thingos::task::Task =
        crate::task::bridge::task_from_thread_state;

    let _: fn(&crate::sched::hooks::ProcessSnapshot) -> thingos::task::Task =
        crate::task::bridge::task_from_snapshot;

    let _: fn(&[crate::sched::state::ThreadState]) -> thingos::job::Job =
        crate::job::bridge::job_from_thread_states;

    let _: fn(&crate::sched::hooks::ProcessSnapshot) -> thingos::job::JobExit =
        crate::job::bridge::job_exit_from_snapshot;

    let _: fn(Option<i32>) -> thingos::job::JobWaitResult =
        crate::job::bridge::job_wait_result_from_poll;

    let _: fn(&crate::sched::hooks::ProcessSnapshot) -> thingos::group::Group =
        crate::group::bridge::group_from_snapshot;

    let _: fn(&crate::sched::hooks::ProcessSnapshot) -> thingos::authority::Authority =
        crate::authority::bridge::authority_from_snapshot;

    let _: fn(&crate::sched::hooks::ProcessSnapshot) -> thingos::place::Place =
        crate::place::bridge::place_from_snapshot;

    let _: fn(KindId, alloc::vec::Vec<u8>) -> thingos::message::Message =
        crate::message::bridge::message_from_parts;
}

#[test]
fn kind_ids_match_kindc_generated_constants() {
    // ── message ───────────────────────────────────────────────────────────────
    assert_eq!(KindId::THINGOS_MESSAGE.0, parse_generated_kind_id("KIND_ID_THINGOS_MESSAGE"));
    assert_eq!(KindId::THINGOS_JOB_EXIT.0, parse_generated_kind_id("KIND_ID_THINGOS_JOB_EXIT"));

    // ── authority ─────────────────────────────────────────────────────────────
    assert_eq!(
        thingos::authority::KIND_ID_THINGOS_AUTHORITY,
        parse_generated_kind_id("KIND_ID_THINGOS_AUTHORITY"),
    );

    // ── task ──────────────────────────────────────────────────────────────────
    assert_eq!(
        thingos::task::KIND_ID_THINGOS_TASK,
        parse_generated_kind_id("KIND_ID_THINGOS_TASK"),
    );
    assert_eq!(
        thingos::task::KIND_ID_THINGOS_TASK_STATE,
        parse_generated_kind_id("KIND_ID_THINGOS_TASK_STATE"),
    );

    // ── job ───────────────────────────────────────────────────────────────────
    assert_eq!(
        thingos::job::KIND_ID_THINGOS_JOB,
        parse_generated_kind_id("KIND_ID_THINGOS_JOB"),
    );
    assert_eq!(
        thingos::job::KIND_ID_THINGOS_JOB_STATE,
        parse_generated_kind_id("KIND_ID_THINGOS_JOB_STATE"),
    );
    assert_eq!(
        thingos::job::KIND_ID_THINGOS_JOB_EXIT,
        parse_generated_kind_id("KIND_ID_THINGOS_JOB_EXIT"),
    );
    assert_eq!(
        thingos::job::KIND_ID_THINGOS_JOB_WAIT_RESULT,
        parse_generated_kind_id("KIND_ID_THINGOS_JOB_WAIT_RESULT"),
    );

    // ── group ─────────────────────────────────────────────────────────────────
    assert_eq!(
        thingos::group::KIND_ID_THINGOS_GROUP,
        parse_generated_kind_id("KIND_ID_THINGOS_GROUP"),
    );

    // ── place ─────────────────────────────────────────────────────────────────
    assert_eq!(
        thingos::place::KIND_ID_THINGOS_PLACE,
        parse_generated_kind_id("KIND_ID_THINGOS_PLACE"),
    );
}
