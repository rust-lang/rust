extern crate core as bäz; //~ ERROR non-ascii idents

use föö::bar; //~ ERROR non-ascii idents

mod föö { //~ ERROR non-ascii idents
    pub fn bar() {}
}

fn bär( //~ ERROR non-ascii idents
    bäz: isize //~ ERROR non-ascii idents
    ) {
    let _ö: isize; //~ ERROR non-ascii idents

    match (1, 2) {
        (_ä, _) => {} //~ ERROR non-ascii idents
    }
}

struct Föö { //~ ERROR non-ascii idents
    föö: isize //~ ERROR non-ascii idents
}

enum Bär { //~ ERROR non-ascii idents
    Bäz { //~ ERROR non-ascii idents
        qüx: isize //~ ERROR non-ascii idents
    }
}

extern "C" {
    fn qüx();  //~ ERROR non-ascii idents
}

fn main() {}
