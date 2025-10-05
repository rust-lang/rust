struct X {}

fn main() {
    let mut __daikon_nonce = 0;
    let mut __unwrap_nonce = NONCE_COUNTER.lock().unwrap();
    __daikon_nonce = *__unwrap_nonce;
    *__unwrap_nonce += 1;
    drop(__unwrap_nonce);
    dtrace_entry("main:::ENTER", __daikon_nonce);
    dtrace_newline();
    dtrace_exit("main:::EXIT1", __daikon_nonce);
    dtrace_newline();
    return;
}

impl X<> {
    pub fn dtrace_print_fields(&self, depth: i32, prefix: String) {
        if depth == 0 { return; }
    }
    pub fn dtrace_print_fields_vec(v: &Vec<&X>, depth: i32, prefix: String) {
        if depth == 0 { return; }
    }
}
