// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::{Debug, Formatter, Error};
use std::collections::HashMap;

trait HasInventory {
    fn getInventory<'s>(&'s self) -> &'s mut Inventory;
    fn addToInventory(&self, item: &Item);
    fn removeFromInventory(&self, itemName: &str) -> bool;
}

trait TraversesWorld {
    fn attemptTraverse(&self, room: &Room, directionStr: &str) -> Result<&Room, &str> {
        let direction = str_to_direction(directionStr);
        let maybe_room = room.direction_to_room.get(&direction);
        //~^ ERROR cannot infer an appropriate lifetime for autoref due to conflicting requirements
        match maybe_room {
            Some(entry) => Ok(entry),
            _ => Err("Direction does not exist in room.")
        }
    }
}


#[derive(Debug, Eq, PartialEq, Hash)]
enum RoomDirection {
    West,
    East,
    North,
    South,
    Up,
    Down,
    In,
    Out,

    None
}

struct Room {
    description: String,
    items: Vec<Item>,
    direction_to_room: HashMap<RoomDirection, Room>,
}

impl Room {
    fn new(description: &'static str) -> Room {
        Room {
            description: description.to_string(),
            items: Vec::new(),
            direction_to_room: HashMap::new()
        }
    }

    fn add_direction(&mut self, direction: RoomDirection, room: Room) {
        self.direction_to_room.insert(direction, room);
    }
}

struct Item {
    name: String,
}

struct Inventory {
    items: Vec<Item>,
}

impl Inventory {
    fn new() -> Inventory {
        Inventory {
            items: Vec::new()
        }
    }
}

struct Player {
    name: String,
    inventory: Inventory,
}

impl Player {
    fn new(name: &'static str) -> Player {
        Player {
            name: name.to_string(),
            inventory: Inventory::new()
        }
    }
}

impl TraversesWorld for Player {
}

impl Debug for Player {
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), Error> {
        formatter.write_str("Player{ name:");
        formatter.write_str(&self.name);
        formatter.write_str(" }");
        Ok(())
    }
}

fn str_to_direction(to_parse: &str) -> RoomDirection {
    match to_parse { //~ ERROR match arms have incompatible types
    //~^ expected enum `RoomDirection`, found enum `std::option::Option`
    //~| expected type `RoomDirection`
    //~| found type `std::option::Option<_>`
        "w" | "west" => RoomDirection::West,
        "e" | "east" => RoomDirection::East,
        "n" | "north" => RoomDirection::North,
        "s" | "south" => RoomDirection::South,
        "in" => RoomDirection::In,
        "out" => RoomDirection::Out,
        "up" => RoomDirection::Up,
        "down" => RoomDirection::Down,
        _ => None //~ NOTE match arm with an incompatible type
    }
}

fn main() {
    let mut player = Player::new("Test player");
    let mut room = Room::new("A test room");
    println!("Made a player: {:?}", player);
    println!("Direction parse: {:?}", str_to_direction("east"));
    match player.attemptTraverse(&room, "west") {
        Ok(_) => println!("Was able to move west"),
        Err(msg) => println!("Not able to move west: {}", msg)
    };
}
