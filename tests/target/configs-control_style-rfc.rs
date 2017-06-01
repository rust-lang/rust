// rustfmt-control_style: Rfc

// #1618
fn main() {
    loop {
        if foo {
            if ((right_paddle_speed < 0.) &&
                 (right_paddle.position().y - paddle_size.y / 2. > 5.)) ||
                ((right_paddle_speed > 0.) &&
                     (right_paddle.position().y + paddle_size.y / 2. < game_height as f32 - 5.))
            {
                foo
            }
            if ai_timer.elapsed_time().as_microseconds() > ai_time.as_microseconds() {
                if ball.position().y + ball_radius >
                    right_paddle.position().y + paddle_size.y / 2.
                {
                    foo
                }
            }
        }
    }
}
