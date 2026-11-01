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

fn clone_thing4(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let nc = nc.clone();
    //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    let nc2 = nc;
    nc2
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}

impl NotClone {
    fn other_fn(&self) {}
    fn get_ref_notclone(&self) -> &Self {
        self
    }
}

fn clone_thing5(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let nc = nc.clone();
    //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    let nc2 = nc;
    nc2.other_fn();
    let nc3 = nc2;
    nc3
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}

fn clone_thing6(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let (ret, _) = (nc.clone(), 1);
    //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    let _ = nc.clone();
    ret
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}

fn clone_thing7(nc: Vec<&NotClone>) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let ret = nc[0].clone();
    //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    ret
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}

fn clone_thing8(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let ret = {
        let a = nc.clone();
        //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
        a
    };
    ret
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}

fn clone_thing9(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let cl = || nc.clone();
    //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    let ret = cl();
    ret
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}

fn clone_thing10(nc: &NotClone) -> (NotClone, NotClone) {
    let (a, b) = {
        let a = nc.clone();
        //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
        (a, nc.clone())
        //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
    };
    (a, b)
    //~^ ERROR mismatched type
    //~| ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
    //~| NOTE expected `NotClone`, found `&NotClone`
}

fn clone_thing11(nc: &NotClone) -> NotClone {
    //~^ NOTE expected `NotClone` because of return type
    let a = {
        let nothing = nc.clone();
        let a = nc.clone();
        //~^ NOTE `NotClone` does not implement `Clone`, so `&NotClone` was cloned instead
        let nothing = nc.clone();
        a
    };
    a
    //~^ ERROR mismatched type
    //~| NOTE expected `NotClone`, found `&NotClone`
}
