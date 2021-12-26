fn foo() {
    match () {
        _ => (),
        #![doc("Not allowed here")]
        _ => (),
    }

    match () {
        _ => (),
        _ => (),
        #![doc("Nor here")]
    }

    match () {
        #[cfg(test)]
        #![doc("Nor here")]
        _ => (),
        _ => (),
    }
}
