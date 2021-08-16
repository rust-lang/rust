pub fn a() -> String {
    format!("a_basement_core")
}

pub fn a_addr() -> usize { a as fn() -> String as *const u8 as usize }
