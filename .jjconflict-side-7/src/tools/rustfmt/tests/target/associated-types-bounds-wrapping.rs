// Test proper wrapping of long associated type bounds

pub trait HttpService {
    type WsService: 'static
        + Service<Request = WsCommand, Response = WsResponse, Error = ServerError>;
}
