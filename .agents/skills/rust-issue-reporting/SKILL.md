---
name: rust-issue-reporting
description: Prepare evidence for rust-lang/rust bug reports and maintainer follow-ups, or privately review a human-written report. Use for Rust compiler, standard-library, rustdoc, or bootstrap reporting tasks. Not for implementing fixes, authoring public prose, or submitting reports; reporting permissions belong to the repository's AGENTS.md.
---

# Rust issue reporting

Help the user understand and verify the problem, without writing the issue for them.
Decide whether a maintainer response is warranted before preparing reporting evidence.
For an independent private review, use [the review guidance](#review-and-hand-off-privately), not a new investigation.

## Start with repository policy

Use [AGENTS.md](../../../AGENTS.md#llm-usage-policy) for reporting permissions, prohibited text, and how to handle requests outside those permissions.
This skill supplies the reporting workflow for authorized evidence preparation and private review.

## Establish ownership

Use the established project and problem; ask which repository only when ownership remains unclear.
Read [external repositories](../../../src/doc/rustc-dev-guide/src/external-repos.md)
and [reporting destinations](../../../.github/ISSUE_TEMPLATE/config.yml) to identify the owner and reporting route.
For a suspected security vulnerability, follow the [security reporting instructions](https://www.rust-lang.org/policies/security),
not the public issue workflow.

Apply the [external-repository routing rule](../../../AGENTS.md#external-repositories) before investigating a tool report.
At the destination, read that project's reporting instructions and LLM policy.
Distinguish a Cargo invocation exposing a compiler ICE from a bug in Cargo itself.
The issue chooser routes usage questions to the user forum and language feature requests to Internals.

## Decide what needs a maintainer response

Use reproduction, compiler identity, version comparisons, and existing reports to decide what action is warranted.
Collect the evidence needed for that decision, not every field a possible report would require.
Search open and closed issues and pull requests for the failure, component, and distinctive errors or function names.
For diagnostics, include the error code and relevant language feature; for ICEs, include the panic message and relevant stack frame.
Compare plausible matches' reproductions and compiler versions; read resolutions and maintainer comments.
Distinguish related failures from duplicates; link related reports and explain their connection.
Report an unavailable or unsuccessful search as a limit, not proof that nobody reported the problem.

Choose and explain one outcome:

- **No response needed:** the failure is fixed, a report covers the evidence without a useful addition, or documented guidance resolves the problem.
- **Existing report:** new evidence could change its diagnosis, resolution, or requested action.
- **New issue:** a distinct unresolved failure warrants a response.
- **Undetermined:** identify the fact that could change the recommendation; check it or ask a bounded question.

Unknown fields in an unused template do not block an otherwise supported decision.

If a verified newer beta or nightly no longer exhibits the failure with the same input and options, recommend against a new issue and name the comparison.
An older stable release still failing does not justify continued report preparation.
Distinguish this upstream fix from a workaround that changes the input or configuration.
Further maintainer action needs verified new evidence or an explicit unmet requirement, such as a needed backport; explain it and prefer the existing report.

Separate required behavior from the preferred mechanism and downstream design choices; identify who imposed each constraint.
Check supported configurations and public APIs to establish what each workaround actually solves.
Tie remaining insufficiencies to requirements and evidence; reconsider revisable downstream choices before calling them compiler limitations.
Resolve missing facts through a small authorized check or a bounded question.
A workaround requiring consumer patches or a fork is not a capability of the unchanged compiler or tool.

For example, replacing a language feature with different source code may let a program build without satisfying a requirement to compile unchanged code.
Likewise, a nightly-only workaround does not meet a requirement to support stable Rust.
Establish whether those requirements actually apply before accepting or rejecting the workaround.
When an upstream change remains necessary, identify the minimum needed behavior and compare its work and maintenance cost with the downstream workaround.
This is analysis for the user, not a requirement to design a fix before reporting.

Stop when existing guidance satisfies the user's outcome, not merely when a workaround makes something possible.
For an explicitly unsupported configuration, explain the documented incompatibility and identify the owner of any downstream correction.
A diagnosis or fix proposal is not required to establish a reproducible failure.

## Prepare a warranted reporting handoff

For a warranted issue or follow-up, read the applicable guidance before collecting report-specific evidence.
Revisit the decision as checks return; stop or change the handoff when the earlier action is no longer warranted.
For a follow-up, collect the new information or requested checks, not an entire replacement report.
For a new `rust-lang/rust` issue, choose the template by the observed failure:

- [Compiler bug](../../../.github/ISSUE_TEMPLATE/bug_report.md): incorrect compiler behavior without a more specific form.
- [ICE](../../../.github/ISSUE_TEMPLATE/ice.md): an internal compiler error, including code, compiler identity, error output, and backtrace.
- [Regression](../../../.github/ISSUE_TEMPLATE/regression.md): an unintended change between versions, with a verified working and failing version.
  Do not infer regression status from an ICE or newly discovered failure alone.
- [Diagnostics](../../../.github/ISSUE_TEMPLATE/diagnostics.yaml): problems with compiler error output.
  Capture current output and contrasting cases.
- [Rustdoc](../../../.github/ISSUE_TEMPLATE/rustdoc.md): documentation generation, doctests, search, or rustdoc UI rather than documentation content.
- [Documentation content](../../../.github/ISSUE_TEMPLATE/documentation.yaml): the location and problem in existing documentation.
  Follow its redirects for externally maintained books.
- [Bootstrap](../../../.github/ISSUE_TEMPLATE/bootstrap.md): build-system behavior, with invocation, configuration, OS, source revision, and build log.

Use the remaining templates only when the request actually calls for their issue type.
Do not turn a bug report into a tracking issue or feature proposal.
Read [bug reporting guidance](../../../src/doc/rustc-dev-guide/src/contributing.md#bug-reports) and the selected template.
For that handoff, each requested item and check is required by default: satisfy it, explain why it does not apply, or ask about the blocker.
Missing tools, access, authorization, or time do not make an item inapplicable.
Tell the user if the expected template is unavailable.
Do not import unrelated PR requirements into an evidence-only task.

If no response is needed, stop report preparation and explain the decisive evidence; do not fill or individually waive unused template fields.

## Gather evidence

Use the user's latest corrections and experiments.
Distinguish the user's reports, agent-run checks, and source inspection.
Give exact commands and captured results for agent checks; never say the user ran them or invent checks, logs, or environment details.
Record patches, forks, nonstandard configuration, and prior attempts with their results.

### Reproduction and compiler identity

Reduce to the smallest verified operation retaining the failure; keep the original application when its behavior matters.
Remove unrelated setup and update prerequisites after each successful reduction.
Preserve useful working controls and the actual invocation, including edition, target, crate type, relevant flags, and environment.
For Cargo-based reproductions, retain the manifest, lockfile, features, target configuration, build scripts, and proc-macro or other dependencies when needed.
Do not reduce to a standalone `.rs` file if doing so removes the failure.
Distinguish tested steps from proposals and runnable reproducers from illustrations with placeholders or omitted setup.
For machine-specific identifiers, provide a discovery command and explain which result to select rather than relying on an unexplained local value.
Name required hardware, unavailable dependencies, and inaccessible infrastructure.
Retain useful failing artifacts even before minimization; do not claim an unreduced artifact is minimal or portable.

Record `rustc --version --verbose` for each tested compiler and the command selecting that compiler.
Identify the executable actually used by the failing command, not merely the default compiler installed on the machine.
Record installation provenance: rustup distribution toolchain, distribution package and package version, or local source build.
Mark unknown provenance or relevant runtime dependencies explicitly.
Capture relevant toolchain overrides, compiler wrappers, and environment so the invocation does not silently select a different compiler.
Distinguish release channel, dated nightly, installed toolchain, and compiler built from this checkout.
For local builds, record the full source revision, local patches, build stage, selected binary, and sysroot when relevant.
Use the [build-and-run guide](../../../src/doc/rustc-dev-guide/src/building/how-to-build-and-run.md) for selecting the built compiler.
Record relevant target and host differences; a host triple alone does not identify a cross-compilation target.

For rustdoc, collect its version and, for browser failures, browser identity, version, and useful screenshots.
Include the generated or hosted page identity, exact query or navigation sequence, and cursor position or keystrokes when they affect the failure.
Distinguish console output, generated HTML, browser behavior, and rustdoc JSON.
For bootstrap, collect the source revision, relevant configuration, complete build log, and local modifications without exposing secrets.

For a warranted reporting handoff, follow the applicable instructions' channel checks.
For example, the compiler bug template requests checking beta or nightly when the failure occurs on stable.
For a nightly failure that still needs reporting, the dev-guide requests checking the latest nightly before filing.
Verify freshness from current first-party release or distribution metadata; a toolchain named `nightly` may be stale.
Do not propose an upgrade before comparing the tested version with current metadata.
Record every channel actually tested and its result; never convert source inspection into a runtime comparison.
Report whether any requested development-build check was actually performed, separately from checking a distributed nightly.
For a regression, keep the latest verified working and earliest verified failing versions distinct from a confirmed regression range.
Do not start a full bisection merely because a regression is suspected.

For an ICE or crash, capture the complete error and the backtrace requested by the selected template.
Set the backtrace variable on the reproducing command.
For all report types, preserve diagnostics and logs verbatim; label excerpts and output from different compiler versions.
An ICE establishes a compiler failure, not by itself a soundness bug or its root cause.

### Execution and source limits

Check template invocation examples against the [repository's command requirements](../../../AGENTS.md#repository-guidance) before running them.

Attempt authorized checks needed to support the decision or applicable reporting handoff, even when another cause seems more likely.
Do not treat checks from an unused template as prerequisites for a no-action recommendation.
If a required check cannot run, stop conclusions depending on it and ask the user about the exact missing check, its blocker, why the result matters, and the smallest way forward.
Continue independent authorized work while waiting.
Marking a required check unverified does not complete it.

Inspect source before making source-based claims.
Keep an unperformed runtime comparison separate from a source-based expectation.
Compare local modifications with upstream when attribution matters.
Use full-commit, line-specific permalinks for inspected source, not branch links or abbreviated hashes.
Do not add source analysis, compiler builds, or source links as prerequisites when the reporting requirements and claims do not need them.

## Review and hand off privately

Lead a reporting answer with the recommendation and its evidence.
Explicitly recommend against a new issue when unwarranted; do not append generic filing instructions.
Mention further reporting only when concrete new evidence or an unmet requirement could change that decision.

Keep commands, input files, raw output, version identities, links, and uncertainty distinguishable from interpretation.
For an independently requested private review, compare the human-authored report with the selected template and latest verified evidence.
Check required fields, headings, order, and required text against the selected Markdown template or form rather than imposing a generic report layout.
For a substantial review, assess whether the report separates expected behavior, the observed failure, its consequence, and any established cause.
Check that motivation and upstream versus downstream responsibility are supported, and that execution boundaries identify the actual processes and environments.
A maintainer may recommend keeping a solution downstream without providing an alternative.
For bounded corrections, review the affected passage and its supporting evidence rather than repeating an unrelated whole-report review.

Inspect actual attachments and archive members for paths, hostnames, environment values, and private content.
Preserve raw evidence; redact labeled copies and reinspect the final artifacts.
Tell the user about remaining identifying metadata and inconsistencies between attachments and the human-authored report.
Do not expose private Zulip material through a public handoff.

Before handing off, check commands against actual runs and source links against the inspected revisions and lines.
Check every research, clean-environment, latest-version, and channel-comparison attestation against its supporting evidence.
Keep quoted diagnostics and logs verbatim; verify excerpt labels and compiler identities.
Apply the [checklist](#prepare-a-warranted-reporting-handoff) to warranted reporting handoffs, including evidence-only ones, and requested report reviews—not no-action explanations.
If the user chooses to proceed with an unresolved requirement, mark missing information as unknown and explain which conclusions remain unresolved.
Do not describe the handoff as complete merely because outstanding checks are listed.
Incorporate later results without inventing attestations; say briefly when no material checks remain.
Use the user's requested evidence destination and return the actual artifact paths.
For an LLM-discovered bug, the [policy's bug-discovery conditions](https://forge.rust-lang.org/policies/llm-usage.html#-allowed-with-caveats)
require personal verification before public reporting.
Use [AGENTS.md](../../../AGENTS.md#before-pushing) for disclosure requirements.
