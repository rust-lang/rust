pub struct Def {
    pub id: i32,
}

impl Def {
    pub fn new(id: i32) -> Def {
        Def { id: id }
    }
}

#[no_mangle]
pub fn user() {
    let _ = Def::new(0);
}
