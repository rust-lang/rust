fn main() {
    // 5042 deals with trailing commas, not the indentation issue of these comments
    // When a future PR fixes the inentation issues these test can be updated
    let _ = std::ops::Add::add(10, 20
        // ...
        // ...,
        );

    let _ = std::ops::Add::add(10, 20
        /* ... */
        // ...,
        );

    let _ = std::ops::Add::add(10, 20
        // ...,
        // ...,
        );

    let _ = std::ops::Add::add(10, 20
        // ...,
        /* ...
        */,
        );
}
