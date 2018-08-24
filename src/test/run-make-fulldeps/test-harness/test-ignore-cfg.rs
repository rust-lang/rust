#[test]
#[cfg_attr(ignorecfg, ignore)]
fn shouldignore() {
}

#[test]
#[cfg_attr(noignorecfg, ignore)]
fn shouldnotignore() {
}
