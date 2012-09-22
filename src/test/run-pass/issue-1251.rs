#[link(name = "get_task_id")];

extern mod rustrt {
    #[legacy_exports];
      fn get_task_id() -> libc::intptr_t;
}

fn main() { }
