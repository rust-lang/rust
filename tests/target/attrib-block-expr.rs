fn issue_2073() {
    let x = {
        #![my_attr]
        do_something()
    };

    let x = #[my_attr]
    {
        do_something()
    };

    let x = #[my_attr]
    {};

    {
        #![just_an_attribute]
    };

    let z = #[attr1]
    #[attr2]
    {
        body()
    };

    x = |y| {
        #![inner]
    };

    x = |y| #[outer]
    {};

    x = |y| {
        //! ynot
    };

    x = |y| #[outer]
    unsafe {};

    let x = unsafe {
        #![my_attr]
        do_something()
    };

    let x = #[my_attr]
    unsafe {
        do_something()
    };

    // This is a dumb but possible case
    let x = #[my_attr]
    unsafe {};

    x = |y| #[outer]
    #[outer2]
    unsafe {
        //! Comment
    };
}
