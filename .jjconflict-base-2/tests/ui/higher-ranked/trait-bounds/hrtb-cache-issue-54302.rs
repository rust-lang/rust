// Regression test for #54302.
//
// We were incorrectly using the "evaluation cache" (which ignored
// region results) to conclude that `&'static str: Deserialize`, even
// though it would require that `for<'de> 'de: 'static`, which is
// clearly false.

trait Deserialize<'de> {}

trait DeserializeOwned: for<'de> Deserialize<'de> {}
impl<T> DeserializeOwned for T where T: for<'de> Deserialize<'de> {}

// Based on this impl, `&'static str` only implements Deserialize<'static>.
// It does not implement for<'de> Deserialize<'de>.
impl<'de: 'a, 'a> Deserialize<'de> for &'a str {}

fn main() {
    fn assert_deserialize_owned<T: DeserializeOwned>() {}
    assert_deserialize_owned::<&'static str>(); //~ ERROR

    // It correctly does not implement for<'de> Deserialize<'de>.
    // fn assert_hrtb<T: for<'de> Deserialize<'de>>() {}
    // assert_hrtb::<&'static str>();
}
