fn main() {
    let internal_code: u32;
    let is_admin = true;
    let access_level = 2;
    if is_admin {
        if access_level == 1 {
            internal_code = 101;
        } else if access_level == 2 {
            println!("Admin access pending for code: {internal_code}"); //~ ERROR E0381
        } else {
            internal_code = 103;
        }
    } else {
        internal_code = 404;
    }
}
