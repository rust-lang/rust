pub struct CrateId {
    local_path: String,
    junk: String,
}

impl CrateId {
    fn new(s: &str) -> CrateId {
        CrateId { local_path: s.to_string(), junk: "wutevs".to_string() }
    }
}

pub fn remove_package_from_database() {
    let mut lines_to_use: Vec<&CrateId> = Vec::new();
    //~^ NOTE `lines_to_use` declared here, outside of the closure body
    let push_id = |installed_id: &CrateId| {
        //~^ NOTE `installed_id` is a reference that is only valid in the closure body
        lines_to_use.push(installed_id);
        //~^ ERROR borrowed data escapes outside of closure
        //~| NOTE `installed_id` escapes the closure body here
        //~| NOTE requirement occurs because of a mutable reference to `Vec<&CrateId>`
        //~| NOTE mutable references are invariant over their type parameter
    };
    list_database(push_id);

    for l in &lines_to_use {
        println!("{}", l.local_path);
    }
}

pub fn list_database<F>(mut f: F)
where
    F: FnMut(&CrateId),
{
    let stuff = ["foo", "bar"];

    for l in &stuff {
        f(&CrateId::new(*l));
    }
}

pub fn main() {
    remove_package_from_database();
}
