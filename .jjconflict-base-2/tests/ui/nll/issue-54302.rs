trait Deserialize<'de> {}

trait DeserializeOwned: for<'de> Deserialize<'de> {}
impl<T> DeserializeOwned for T where T: for<'de> Deserialize<'de> {}

// Based on this impl, `&'static str` only implements Deserialize<'static>.
// It does not implement for<'de> Deserialize<'de>.
impl<'de: 'a, 'a> Deserialize<'de> for &'a str {}

fn main() {
    // Then why does it implement DeserializeOwned? This compiles.
    fn assert_deserialize_owned<T: DeserializeOwned>() {}
    assert_deserialize_owned::<&'static str>();
    //~^ ERROR not general enough

    // It correctly does not implement for<'de> Deserialize<'de>.
    //fn assert_hrtb<T: for<'de> Deserialize<'de>>() {}
    //assert_hrtb::<&'static str>();
}
