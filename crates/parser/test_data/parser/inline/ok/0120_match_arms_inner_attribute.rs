fn foo() {
    match () {
        #![doc("Inner attribute")]
        #![doc("Can be")]
        #![doc("Stacked")]
        _ => (),
    }
}
