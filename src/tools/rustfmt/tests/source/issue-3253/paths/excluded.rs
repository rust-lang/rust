// This module is not imported in the cfg_if macro in lib.rs so it is ignored
// while the foo and bar mods are formatted.
// Check the corresponding file in tests/target/issue-3253/paths/excluded.rs
trait CoolerTypes { fn dummy(&self) {
}
}
