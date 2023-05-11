struct NotClone;

fn main() {
    clone_thing(&NotClone);
}

fn clone_thing(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    nc.clone()
    //~^ ERROR mismatched type
    //~| NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    //~| NOTE expected `NotClone`, found `&NotClone`
}
