# display_proto_tests

Test harness crate for display protocol conformance and negotiation scenarios.

## Running
- ABI protocol tests: `cargo test -p abi`
- Negotiation scenarios: `cargo test -p display_proto_tests`

The integration tests use an in-memory link and simulated driver/client behavior
so no real GPU or userspace driver is required.
