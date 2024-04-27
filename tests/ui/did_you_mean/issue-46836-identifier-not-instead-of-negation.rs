fn gratitude() {
    let for_you = false;
    if not for_you {
        //~^ ERROR unexpected `for_you` after identifier
        println!("I couldn't");
    }
}

fn qualification() {
    let the_worst = true;
    while not the_worst {
        //~^ ERROR unexpected `the_worst` after identifier
        println!("still pretty bad");
    }
}

fn should_we() {
    let not = true;
    if not  // lack of braces is [sic]
        println!("Then when?");
    //~^ ERROR expected `{`, found `;
    //~| ERROR unexpected `println` after identifier
}

fn sleepy() {
    let resource = not 2;
    //~^ ERROR unexpected `2` after identifier
}

fn main() {
    let be_smothered_out_before = true;
    let young_souls = not be_smothered_out_before;
    //~^ ERROR unexpected `be_smothered_out_before` after identifier
}
