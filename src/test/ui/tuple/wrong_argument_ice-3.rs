struct Process;

pub type Group = (Vec<String>, Vec<Process>);

fn test(process: &Process, groups: Vec<Group>) -> Vec<Group> {
    let new_group = vec![String::new()];

    if groups.capacity() == 0 {
        groups.push(new_group, vec![process]);
        //~^ ERROR this function takes 1 argument but 2 arguments were supplied
        return groups;
    }

    todo!()
}

fn main() {}
