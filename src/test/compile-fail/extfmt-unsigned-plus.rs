// error-pattern:only valid in signed #ifmt conversion

fn main() {
    // Can't use a sign on unsigned conversions
    #ifmt["%+u", 10u];
}
