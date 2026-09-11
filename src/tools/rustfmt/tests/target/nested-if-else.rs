fn issue1518() {
    Some(Object {
        field: if a {
            a_thing
        } else if b {
            b_thing
        } else {
            c_thing
        },
    })
}
