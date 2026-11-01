//! Test error handling for undefined macro calls

fn main() {
    iamnotanextensionthatexists!("");
    //~^ ERROR cannot find macro `iamnotanextensionthatexists` in this scope
}
