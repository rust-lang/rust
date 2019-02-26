#[derive(Copy, Clone, Debug, PartialEq)]
struct BadNumErr;

fn try_num(x: i32) -> Result<i32, BadNumErr> {
    if x <= 5 {
        Ok(x + 1)
    } else {
        Err(BadNumErr)
    }
}

type ResOpt = Result<Option<i32>, BadNumErr>;
type OptRes = Option<Result<i32, BadNumErr>>;

fn main() {
    let mut x: ResOpt = Ok(Some(5));
    let mut y: OptRes = Some(Ok(5));
    assert_eq!(x, y.transpose());
    assert_eq!(x.transpose(), y);

    x = Ok(None);
    y = None;
    assert_eq!(x, y.transpose());
    assert_eq!(x.transpose(), y);

    x = Err(BadNumErr);
    y = Some(Err(BadNumErr));
    assert_eq!(x, y.transpose());
    assert_eq!(x.transpose(), y);

    let res: Result<Vec<i32>, BadNumErr> =
        (0..10)
            .map(|x| {
                let y = try_num(x)?;
                Ok(if y % 2 == 0 {
                    Some(y - 1)
                } else {
                    None
                })
            })
            .filter_map(Result::transpose)
            .collect();

    assert_eq!(res, Err(BadNumErr))
}
