// rustfmt-indent_style: Block

// #1618
fn main() {
    loop {
        if foo {
            if ((right_paddle_speed < 0.) && (right_paddle.position().y - paddle_size.y / 2. > 5.))
                || ((right_paddle_speed > 0.)
                    && (right_paddle.position().y + paddle_size.y / 2. < game_height as f32 - 5.))
            {
                foo
            }
            if ai_timer.elapsed_time().as_microseconds() > ai_time.as_microseconds() {
                if ball.position().y + ball_radius > right_paddle.position().y + paddle_size.y / 2.
                {
                    foo
                }
            }
        }
    }
}

fn issue1656() {
    {
        {
            match rewrite {
                Some(ref body_str)
                    if (!body_str.contains('\n') && body_str.len() <= arm_shape.width)
                        || !context.config.match_arm_blocks()
                        || (extend && first_line_width(body_str) <= arm_shape.width)
                        || is_block =>
                {
                    return None;
                }
                _ => {}
            }
        }
    }
}
