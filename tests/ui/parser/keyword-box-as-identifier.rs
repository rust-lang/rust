fn main() {
    let box = 0;
    //~^ ERROR expected pattern, found `=`
    let box: bool;
    //~^ ERROR expected pattern, found `:`
    let mut box = 0;
    //~^ ERROR expected pattern, found `=`
    let (box,) = (0,);
    //~^ ERROR expected pattern, found `,`
}
