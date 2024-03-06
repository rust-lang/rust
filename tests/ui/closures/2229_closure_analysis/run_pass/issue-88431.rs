//@ edition:2021
//@ check-pass

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

pub struct GameMode {}

struct GameStateManager<'a> {
    gamestate_stack: Vec<Box<dyn GameState<'a> + 'a>>,
}

pub trait GameState<'a> {}

async fn construct_gamestate_replay<'a>(
    _gamemode: &GameMode,
    _factory: &mut GameStateManager<'a>,
) -> Box<dyn GameState<'a> + 'a> {
    unimplemented!()
}

type FutureGameState<'a, 'b> = Pin<Box<dyn Future<Output = Box<dyn GameState<'a> + 'a>> + 'b>>;

struct MenuOption<'a> {
    command: Box<dyn for<'b> Fn(&'b mut GameStateManager<'a>) -> FutureGameState<'a, 'b> + 'a>,
}

impl<'a> MenuOption<'a> {
    fn new(
        _command: impl for<'b> Fn(&'b mut GameStateManager<'a>) -> FutureGameState<'a, 'b> + 'a,
    ) -> Self {
        unimplemented!()
    }
}

struct MenuState<'a> {
    options: Vec<MenuOption<'a>>,
}

impl<'a> GameState<'a> for MenuState<'a> {}

pub async fn get_replay_menu<'a>(
    gamemodes: &'a HashMap<&str, GameMode>,
) -> Box<dyn GameState<'a> + 'a> {
    let recordings: Vec<String> = vec![];
    let _ = recordings
        .into_iter()
        .map(|entry| {
            MenuOption::new(move |f| {
                Box::pin(construct_gamestate_replay(&gamemodes[entry.as_str()], f))
            })
        })
        .collect::<Vec<_>>();

    todo!()
}

fn main() {}
