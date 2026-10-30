macro_rules! identity { ($( $tt:tt )*) => { $( $tt )* } }

fn main() {
    #[expect(wat::doesnt_exist)]
    //~^ WARN cannot be used on
    //~| WARN previously accepted
    //~| ERROR unknown tool name
    println!();

    identity!{
        #[expect(wat::doesnt_exist)]
        //~^ WARN cannot be used on
        //~| WARN previously accepted
        //~| ERROR unknown tool name
        println!();
    }
}
