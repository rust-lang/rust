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

fn clone_thing2(nc: &NotClone) -> NotClone {
    let nc: NotClone = nc.clone();
    //~^ ERROR mismatched type
    //~| NOTE expected due to this
    //~| NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    //~| NOTE expected `NotClone`, found `&NotClone`
    nc
}

fn clone_thing3(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let nc = nc.clone();
    //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    nc
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}