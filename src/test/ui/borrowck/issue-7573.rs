pub struct CrateId {
    local_path: String,
    junk: String
}

impl CrateId {
    fn new(s: &str) -> CrateId {
        CrateId {
            local_path: s.to_string(),
            junk: "wutevs".to_string()
        }
    }
}

pub fn remove_package_from_database() {
    let mut lines_to_use: Vec<&CrateId> = Vec::new();
        //~^ NOTE cannot infer an appropriate lifetime
    let push_id = |installed_id: &CrateId| {
        //~^ NOTE borrowed data cannot outlive this closure
        //~| NOTE ...so that variable is valid at time of its declaration
        lines_to_use.push(installed_id);
        //~^ ERROR borrowed data cannot be stored outside of its closure
        //~| NOTE cannot be stored outside of its closure
    };
    list_database(push_id);

    for l in &lines_to_use {
        println!("{}", l.local_path);
    }

}

pub fn list_database<F>(mut f: F) where F: FnMut(&CrateId) {
    let stuff = ["foo", "bar"];

    for l in &stuff {
        f(&CrateId::new(*l));
    }
}

pub fn main() {
    remove_package_from_database();
}
