mod tests {
    #[test]
<<<<<<< HEAD
//~^ ERROR encountered diff marker
//~| NOTE between this marker and `=======`

//~| NOTE conflict markers indicate that
//~| HELP if you're having merge conflicts
//~| NOTE for an explanation on these markers

    fn test1() {
=======
//~^ NOTE between this marker and `>>>>>>>`
    fn test2() {
>>>>>>> 7a4f13c blah blah blah
//~^ NOTE this marker concludes the conflict region
    }
}
