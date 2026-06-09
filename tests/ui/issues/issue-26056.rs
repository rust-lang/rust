trait MapLookup<Q> {
    type MapValue;
}

impl<K> MapLookup<K> for K {
    type MapValue = K;
}

trait Map: MapLookup<<Self as Map>::Key> {
    type Key;
}

impl<K> Map for K {
    type Key = K;
}


fn main() {
    let _ = &()
        as &dyn Map<Key=u32,MapValue=u32>;
    //~^ ERROR E0038
}
