fn main() {
    let mut map1 = hashbrown::HashMap::new();
    map1.insert(1u8, "");
    map1.reserve(1000);
    let mut map2 = hashbrown::HashMap::new();
    map2.insert(1i16, "");
    map2.reserve(1000);
    let mut map3 = hashbrown::HashMap::new();
    map3.insert(3u16, "");
    map3.reserve(1000);
    let mut map4 = hashbrown::HashMap::new();
    map4.insert(3u64, "");
    map4.reserve(1000);
    dbg!((
        map1.iter().next(),
        map2.iter().next(),
        map3.iter().next(),
        map4.iter().next()
    ));
}
