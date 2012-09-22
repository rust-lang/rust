// ABI is cdecl by default

extern mod rustrt {
    #[legacy_exports];
    fn get_task_id() -> libc::intptr_t;
}

fn main() {
    rustrt::get_task_id();
}
