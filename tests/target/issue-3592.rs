fn r() -> (Biz, ()) {
    (
        Biz {
            #![cfg(unix)]
            field: 9
        },
        Biz {
            #![cfg(not(unix))]
            field: 200
        },
        (),
    )
}
