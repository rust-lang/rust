// Fixes: #121623

fn main() {
    match () {
        _ => 'b: {
            continue 'b;
            //~^ ERROR [E0696]
        }
    }
}
