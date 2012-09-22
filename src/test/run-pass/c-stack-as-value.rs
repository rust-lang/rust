#[abi = "cdecl"]
extern mod rustrt {
    #[legacy_exports];
    fn get_task_id() -> libc::intptr_t;
}

fn main() {
    let _foo = rustrt::get_task_id;
}
