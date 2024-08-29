pub mod unix {
    #[allow(non_camel_case_types)]
    pub type pid_t = i32;
    pub fn get_pid() -> pid_t {
        0
    }
    pub fn set_pid(_: pid_t) {}
}
