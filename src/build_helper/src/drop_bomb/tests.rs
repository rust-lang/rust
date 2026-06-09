use super::DropBomb;

#[test]
#[should_panic]
fn test_arm() {
    let bomb = DropBomb::arm("hi :3");
    drop(bomb); // <- armed bomb should explode when not defused
}

#[test]
fn test_defuse() {
    let mut bomb = DropBomb::arm("hi :3");
    bomb.defuse();
    drop(bomb); // <- defused bomb should not explode
}
