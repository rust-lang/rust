// ignore-test this is not a test

macro_rules! m {
    () => { mod mod_file_not_owning_aux2; }
}
m!();
