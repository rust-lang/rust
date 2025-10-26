//! Regression test for https://github.com/rust-lang/rust/issues/12860

//@ run-pass
use std::collections::HashSet;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct XYZ {
    x: isize,
    y: isize,
    z: isize
}

fn main() {
    let mut connected = HashSet::new();
    let mut border = HashSet::new();

    let middle = XYZ{x: 0, y: 0, z: 0};
    border.insert(middle);

    while !border.is_empty() && connected.len() < 10000 {
        let choice = *(border.iter().next().unwrap());
        border.remove(&choice);
        connected.insert(choice);

        let cxp = XYZ{x: choice.x + 1, y: choice.y, z: choice.z};
        let cxm = XYZ{x: choice.x - 1, y: choice.y, z: choice.z};
        let cyp = XYZ{x: choice.x, y: choice.y + 1, z: choice.z};
        let cym = XYZ{x: choice.x, y: choice.y - 1, z: choice.z};
        let czp = XYZ{x: choice.x, y: choice.y, z: choice.z + 1};
        let czm = XYZ{x: choice.x, y: choice.y, z: choice.z - 1};

        if !connected.contains(&cxp) {
            border.insert(cxp);
        }
        if  !connected.contains(&cxm){
            border.insert(cxm);
        }
        if !connected.contains(&cyp){
            border.insert(cyp);
        }
        if !connected.contains(&cym) {
            border.insert(cym);
        }
        if !connected.contains(&czp){
            border.insert(czp);
        }
        if !connected.contains(&czm) {
            border.insert(czm);
        }
    }
}
