fn main() {
    let x1 = HashMap::new(); //~ ERROR cannot find
    let x2 = GooMap::new(); //~ ERROR cannot find

    let y1: HashMap; //~ ERROR cannot find type
    let y2: GooMap; //~ ERROR cannot find type
}
