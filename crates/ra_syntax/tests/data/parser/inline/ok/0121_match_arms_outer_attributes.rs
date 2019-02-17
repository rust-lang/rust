fn foo() {
    match () {
        #[cfg(feature = "some")]
        _ => (),
        #[cfg(feature = "other")]
        _ => (),
        #[cfg(feature = "many")]
        #[cfg(feature = "attributes")]
        #[cfg(feature = "before")]
        _ => (),
    }
}
