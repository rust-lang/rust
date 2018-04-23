struct User {
    name: String,
}

fn main() {
    let mut user1 = User{
        name: String::from("Kurtis"),
    };
    mutator_part1(&mut user1);
}

fn mutator_part1(user: &mut User) {
    mutator_part2(&mut user);
}

fn mutator_part2(user: &mut User) {
    user.name = String::from("Steve");
}
