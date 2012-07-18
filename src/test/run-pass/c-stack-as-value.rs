#[abi = "cdecl"]
extern mod rustrt {
    fn get_task_id();
}

fn main() {
    let _foo = rustrt::get_task_id;
}
