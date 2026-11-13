fn main() {
    <
    &[u32]
    >::iter;
    //~^ ERROR no associated function or constant named `iter` found for reference `&[u32]`

    <&[u32]/*c*/>::iter;
    //~^ ERROR no associated function or constant named `iter` found for reference `&[u32]`

    <&[u32] // c
    >::iter;
    //~^ ERROR no associated function or constant named `iter` found for reference `&[u32]`

    </* header
    // out */
    &[u32]>::iter;
    //~^ ERROR no associated function or constant named `iter` found for reference `&[u32]`

    </*/c*/&[u32]>::iter;
    //~^ ERROR no associated function or constant named `iter` found for reference `&[u32]`

    let _ = (
        r#"
// "#,
        <&[u32]>::iter,
        //~^ ERROR no associated function or constant named `iter` found for reference `&[u32]`
    );

    fn foo<'a>() {
        < // */
        &[u32]>::iter;
        //~^ ERROR no associated function or constant named `iter` found for reference `&[u32]`
    }
}
