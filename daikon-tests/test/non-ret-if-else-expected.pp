fn boop(x: i32) -> i32 {
    let mut __daikon_nonce = 0;
    let mut __unwrap_nonce = NONCE_COUNTER.lock().unwrap();
    __daikon_nonce = *__unwrap_nonce;
    *__unwrap_nonce += 1;
    drop(__unwrap_nonce);
    dtrace_entry("boop:::ENTER", __daikon_nonce);
    dtrace_print_prim::<i32>(x, String::from("x"));
    dtrace_newline();
    let y = if x == 12 { 22 } else { 19 };
    dtrace_exit("boop:::EXIT1", __daikon_nonce);
    dtrace_print_prim::<i32>(x, String::from("x"));
    let __daikon_ret: i32 = y;
    dtrace_print_prim::<i32>(__daikon_ret, String::from("return"));
    dtrace_newline();
    return __daikon_ret;
}

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
